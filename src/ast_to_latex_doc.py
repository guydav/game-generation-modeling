import argparse
from collections import defaultdict
from itertools import chain
import tatsu

import ast_printer
from ast_parser import ASTParser, ASTParentMapper
from ast_utils import copy_ast, load_asts, replace_child

parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    # '../dsl/problems-few-objects.pddl',
    # '../dsl/problems-medium-objects.pddl',
    # '../dsl/problems-many-objects.pddl',
    './dsl/interactive-beta.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
DEFAULT_OUTPUT_PATH ='./data/dsl.tex'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)
parser.add_argument('-p', '--print-dsls', action='store_true')





class DSLToLatexParser(ASTParser):
    def __init__(self) -> None:
        super().__init__()
        self.rules_by_section = defaultdict(lambda: defaultdict(list))
        self.processors = []

    def register_processor(self, processor):
        self.processors.append(processor)

    def __call__(self, ast, **kwargs):
        # TODO: set up any local kwargs if need be
        return super().__call__(ast, **kwargs)

    def _handle_tuple(self, ast, **kwargs):
        if len(ast) > 0 and isinstance(ast[0], str) and ast[0].startswith('(:'):
            kwargs['section'] = ast[0][2:]

        return super()._handle_tuple(ast, **kwargs)

    def _handle_ast(self, ast, **kwargs):
        if 'section' in kwargs and ast.parseinfo:
            self.rules_by_section[kwargs['section']][ast.parseinfo.rule].append(ast)

        super()._handle_ast(ast, **kwargs)        

    def process(self):
        for processor in self.processors:
            if isinstance(processor, SectionTranslator):
                if processor.section_key in self.rules_by_section:
                    remaining_keys = processor.process(self.rules_by_section[processor.section_key])
                    if remaining_keys:
                        print(f'After processing section {processor.section_key}, the following keys remain: {remaining_keys}')

                    if processor.unused_rules:
                        print(f'After processing section {processor.section_key}, these rules are unused: {processor.unused_rules}')
                
                else:
                    raise ValueError(f'Found section processor for unknown section key: {processor.section_key}')    

            else:
                raise ValueError(f'Found processor of unknown type: {type(processor)}')

        
UNUSED_RULE_COLOR = 'gray'


class SectionTranslator:
    def __init__(self, section_key, core_blocks, additional_blocks=None, section_name=None, section_type='grammar', unused_rule_color=UNUSED_RULE_COLOR):
        self.section_key = section_key
        self.core_blocks = core_blocks
        self.additional_blocks = additional_blocks if additional_blocks is not None else []
        self.section_name = section_name if section_name is not None else section_key.capitalize()
        self.section_type = section_type
        self.unused_rule_color = unused_rule_color

        self.lines = []
        self.remaining_rules = []
        self.unused_rules = []
        

    def process(self, section_data):
        keys = set(section_data.keys())

        for blocks, is_core in zip((self.core_blocks, self.additional_blocks), (True, False)):
            for block_text, block_rules in blocks:

                if isinstance(block_rules, str):
                    if block_rules not in keys:
                        if not is_core:
                            continue

                        block_text = f'{{ \\color{{{self.unused_rule_color}}} {block_text} }}'
                        self.unused_rules.append(block_rules)
                    else:
                        keys.remove(block_rules)

                    self.lines.append(block_text)

                elif is_core or any([rule in keys for rule in block_rules]):
                    self.lines.extend(block_text.split('\n'))
                    for rule in block_rules:
                        if rule in keys:
                            keys.remove(rule)

                        elif is_core:
                            self.unused_rules.append(rule)

                self.lines.append('')

            self.lines.append('')

        return keys

    def output(self):
        return [
            f'\\subsection{{ {self.section_name} }}',
            f'\\begin{{ {self.section_type} }}'
        ] + self.lines + [f'\\end{{ {self.section_type} }}']


FUNCTION_COMPARISON = 'function_comparison'
VARIABLE_LIST = 'variable_list'
PREDICATE = 'predicate'
SHARED_BLOCKS = {
    FUNCTION_COMPARISON: (r"""<f-comp> ::= (<comp-op> <function-eval-or-number> <function-eval-or-number>) \alt
    (= <function-eval-or-number>$^+$)
    
<comp-op> ::=  \textlangle \ | \textlangle = \ | = \ | \textrangle \ | \textrangle =

<function-eval-or-number> ::= <function-eval> | <number>

<function-eval> ::= (<name> <function-term>$^+$)

<function-term> ::= <name> | <variable> | <number> | <predicate>""", ('function_comparison', 'function_eval')),

    VARIABLE_LIST: (r"""<variable-list> ::= (<variable-type-def>$^+$)

<variable-type-def> ::= <variable>$^+$ - <type-def>

<variable> ::= /\textbackslash?[a-z][a-z0-9]*/  "#" a question mark followed by a letter, optionally followed by additional letters or numbers

<type-def> ::= <name> | <either-types>

<either-types> ::= (either <name>$^+$)""", ('variable_list', 'variable_type_def', 'either_types')),

    PREDICATE: (r"""<predicate> ::= (<name> <predicate-term>$^*$)

<predicate-term> ::= <name> | <variable> | <predicate> "#" In at least one case, I wanted to have a predicate act on other predicates, but I don't know if it makes sense""",
('predicate',)),


}


SETUP_BLOCKS = (
    (r"""<setup> ::= (and <setup> <setup>$^+$) \alt
    (or <setup> <setup>$^+$) \alt
    (not <setup>) \alt
    (exists (<typed list(variable)>) <setup>) \alt
    (forall (<typed list(variable)>) <setup>) \alt
    <setup-statement>""", ('setup_and', 'setup_or', 'setup_not', 'setup_exists', 'setup_forall', 'setup_statement')),

    (r"""<setup-statement> ::= (game-conserved <setup-predicate>) \alt
    (game-optional <setup-predicate>)""", ('setup_game_conserved', 'setup_game_optional',)),

    (r"""<setup-predicate> ::= (and <setup-predidcate>$^+$) \alt
    (or <setup-predicate>$^+$) \alt
    (not <setup-predicate> \alt
    <f-comp> \alt
    <predicate>""", ('setup_and_predicate', 'setup_or_predicate', 'setup_not_predicate')),
)


PREFERENCES_BLOCKS = (
    (r"""<constraints> ::= <pref-def> | (and <pref-def>$^+$)
    
<pref-def> ::= <pref-forall> | <preference> 

<pref-forall> ::= (forall <variable-list> <preference>)
    
<preference> ::= (preference <name> <preference-quantifier>)

<preference-quantifier> ::= (exists (<variable-list>) <preference-body> 
\alt  (forall (<variable-list>) <preference-body>)
\alt <preference-body>) 

<preference-body> ::=  <then> | <at-end> """, ('preference', 'pref_forall', 'pref_body_exists')),

    (r'<at-end> ::= (at-end <pref-predicate>)', 'at_end'), 

    (r"""<then> ::= (then <seq-func> <seq-func>$^+$) 

<seq-func> ::= <once> | <once-measure> | <hold> | <hold-while> | <hold-for> | <hold-to-end>
\alt <forall-seq>""", 'then'),

    (r'<once> ::= (once <pref-predicate>)', 'once'),

    (r'<once-measure> ::= (once <pref-predicate> <f-exp>)', 'once-measure'),

    (r'<hold> ::= (hold <pref-predicate>)', 'hold'),

    (r'<hold-while> ::= (hold-while <pref-predicate> <pref-predicate>$^+$)', 'while_hold'),

    (r'<hold-for> ::= (hold-for <number> <pref-predicate>)', 'hold_for'),

    (r'<hold-to-end> ::= (hold-to-end <pref-predicate>)', 'hold_to_end'),

    (r'<forall-seq> ::= (forall-sequence (<variable-list>) <then>)', 'forall_seq'),

    (r"""<pref-predicate> ::= <pref_predicate_and> \alt
    <pref-predicate-or> \alt
    <pref-predicate-not> \alt
    <pref-predicate-exists> \alt
    <pref-predicate-forall> \alt
    <predicate>
    <f-comp>

<pref-predicate-and> ::= (and <pref-predicate>$^+$)

<pref-predicate-or> ::= (or <pref-predicate>$^+$)

<pref-predicate-not> ::= (not <pref-predicate>)

<pref-predicate-exists> ::= (exists <variable-list> <pref-predicate>)

<pref-predicate-forall> ::= (forall <variable-list> <pref-predicate>)""", ('pref_predicate_and', 'pref_predicate_or', 'pref_predicate_not', 'pref_predicate_exists', ))
)


TERMINAL_BLOCKS = (
    (r"""<terminal> ::= (and <terminal>$^+$) \alt
        (or <terminal>$+$) \alt
        (not <terminal>) \alt
        <terminal-comp>""", ('terminal_and', 'terminal_or', 'terminal_not')),

    (r"""<terminal-comp> ::= (<comp-op> <scoring-expr> <scoring-expr>)
    
    <comp-op> ::=  \textlangle \ | \textlangle = \ | = \ | \textrangle \ | \textrangle =""", 'terminal_comp'),
)


SCORING_BLOCKS = (
    (r"""<scoring> ::= (maximize <scoring-expr>) \alt (minimize <scoring-expr>)""", ('scoring_maximize', 'scoring_minimize')),

    (r"""<scoring-expr> ::= (<multi-op> <scoring-expr>$^+$) \alt
        (<binary-op> <scoring-expr> <scoring-expr>) \alt
        (- <scoring-expr>) \alt
        (total-time) \alt
        (total-score) \alt
        <scoring-comp> \alt
        <preference-eval> 
        
    """, ('scoring_multi_expr', 'scoring_neg_expr')),

    (r"""<scoring-comp> ::=  (<comp-op> <scoring-expr> <scoring-expr>) \alt
        (= <scoring-expr>$^+$)
    """, 'scoring_comp'),

    (r"""<preference-eval> ::=  <count-nonoverlapping> \alt
        <count-once> \alt
        <count-once-per-objects> \alt
        <count-longest> \alt
        <count-shortest> \alt
        <count-increasing-measure> \alt
        <count-unique-positions>
    """, 'preference-eval'),

    (r'<count-nonoverlapping> ::= (count-nonoverlapping <name>)', 'count_nonoverlapping'),
    (r'<count-once> ::= (count-once <name>)', 'count_once'),
    (r'<count-once-per-objects> ::= (count-once-per-objects <name>)', 'count_once_per_objects'),
    (r'<count-longest> ::= (count-longest <name>)', 'count_longest'),
    (r'<count-shortest> ::= (count-shortest <name>)', 'count_shortest'),
    (r'<count-total> ::= (count-total <name>)', 'count_total'),
    (r'<count-increasing-measure> ::= (count-increasing-measure <name>)', 'count_increasing_measure'),
    (r'<count-unique-positions> ::= (count-unique-positions <name>)', 'count_unique_positions'),

    (r"""<pref-name-and-types> ::= <name> <pref-object-type>$^*$ 

    <pref-object-type> ::= : <name>
    """, ('pref_name_and_types', 'pref_object_type')),
)


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar) 

    asts = load_asts(args, grammar_parser, should_print=args.print_dsls)
    parser = DSLToLatexParser()

    setup_translator = SectionTranslator('setup', SETUP_BLOCKS, (SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE]))
    pref_translator = SectionTranslator('constraints', PREFERENCES_BLOCKS, (SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE]), section_name='Preferences')
    terminal_translator = SectionTranslator('terminal', TERMINAL_BLOCKS, None, section_name='Terminal Conditions')
    scoring_translator = SectionTranslator('scoring', SCORING_BLOCKS, None)

    # TODO: cross-section translators for the predicates and types
    # TODO: add the overall game definition section
    # TODO: add the rest of the latex preamble
    # TODO: compile latex to pdf immediately from Python?

    parser.register_processor(setup_translator)
    parser.register_processor(pref_translator)
    parser.register_processor(terminal_translator)
    parser.register_processor(scoring_translator)

    for ast in asts:
        parser(ast)
        # ast_copy = copy_ast(grammar_parser, ast)
        # processed_ast = parser(ast_copy)
        # ast_printer.reset_buffers(False)
        # ast_printer.pretty_print_ast(processed_ast, context=dict())
        # print('\r\n' + '=' * 100 + '\r\n')

    parser.process()

    # for section in parser.rules_by_section:
    #     print(f'{section}: {parser.rules_by_section[section].keys()}\n')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
