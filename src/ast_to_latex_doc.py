import argparse
from collections import defaultdict
from itertools import chain
import tatsu
import shutil
import os

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
DEFAULT_OUTPUT_FILE = './latex/dsl_doc.tex'
parser.add_argument('-o', '--output-file', default=DEFAULT_OUTPUT_FILE)
parser.add_argument('-p', '--print-dsls', action='store_true')
DEFAULT_TEMPLATE_FILE = './latex/template.tex'
parser.add_argument('-f', '--template-file', default=DEFAULT_TEMPLATE_FILE)


DEFAULT_PREFIX_LINES = [
    r'\section{DSL Docuemntation as of \today}'
    r'\subsection{Game Definition}',
    r'\begin{grammar}',
    r'<game> ::= (define (game <name>) \\',
    r'  (:domain <name>) \\',
    r'  (:setup <setup>) \\',
    r'  (:constraints <constraints>) \\',
    r'  (:terminal <terminal>) \\',
    r'  (:scoring <scoring>) \\'
    r')',
    '',
    r'<name> ::= /[A-z]+(_[A-z0-9]+)*/ "#" a letter, optionally followed by letters, numbers, and underscores',
r'\end{grammar}',
'',
]

TEMPLATE_PLACEHOLDER = '{{BODY}}'
class DSLToLatexParser(ASTParser):
    def __init__(self, template_path, output_path, prefix_lines=DEFAULT_PREFIX_LINES, postfix_lines=None) -> None:
        super().__init__()
        self.template_path = template_path
        self.output_path = output_path

        self.rules_by_section = defaultdict(lambda: defaultdict(list))
        self.processors = []
        self.prefix_lines = prefix_lines if prefix_lines is not None else []
        self.postfix_lines = postfix_lines if postfix_lines is not None else []

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

            elif isinstance(processor, RuleTypeTranslator):
                for section_data in self.rules_by_section.values():
                    processor.process(section_data)

            else:
                raise ValueError(f'Found processor of unknown type: {type(processor)}')

    def output(self):
        lines = self.prefix_lines[:]

        for processor in self.processors:
            lines.extend(processor.output())
            lines.append('\n')

        lines.extend(self.postfix_lines)

        template_text = open(self.template_path).read()
        template_text = template_text.replace(TEMPLATE_PLACEHOLDER, '\n'.join(lines))

        open(self.output_path, 'w').write(template_text)


UNUSED_RULE_COLOR = 'gray'
UNDESCRIBED_ELEMENT_COLOR = 'red'

SETUP_SECTION_KEY = 'setup'
PREFERENCES_SECTION_KEY = 'constraints'
TERMINAL_SECTION_KEY = 'terminal'
SCORING_SECTION_KEY = 'scoring'
PREDICATES_SECTION_KEY = 'predicates'
TYPES_SECTION_KEY = 'types'

PRE_NOTES_KEY = 'pre'
POST_NOTES_KEY = 'post'

NOTES = {
    SETUP_SECTION_KEY: {
        PRE_NOTES_KEY: r"""PDDL doesn't have any close equivalent of this, but when reading through the games participants specify, 
        they often require some transformation of the room from its initial state before the game can be played.
        We could treat both as parts of the gameplay, but I thought there's quite a bit to be gained by splitting them -- for example,
        the policies to setup a room are quite different from the policies to play a game (much more static). \\

        The one nuance here came from the (game-conserved ...) and (game-optional ...) elements. It seemed to me that some setup elements should be maintained
        throughout gameplay (for example, if you place a bin somewhere to throw into, it shouldn't move unless specified otherwise).
        Other setup elements can, or often must change -- for example, if you set the balls on the desk to throw them, you'll have to pick them up off the desk to throw them.
        These elements provide that context, which could be useful for verifying that agents playing the game don't violate these conditions.
        """,
    },
    PREFERENCES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""PDDL calls their temporal preferences 'constraints', but that's not entirely the right name for us. Maybe we should rename? \\

        Any syntax elements that are defined (because at some point a game needed them) but are currently unused (in the interactive games) will appear in {{ \color{{{unused_color}}} {unused_color} }}.
        """.format(unused_color=UNUSED_RULE_COLOR)
    },
    TERMINAL_SECTION_KEY: {
        PRE_NOTES_KEY: r"""There's always assumed to be a time limit after which the game is over if nothing else, but some participants specified other terminal conditions.
        """,
        POST_NOTES_KEY: r"""For a full specification of the <scoring-expr> token, see the scoring section below.
        """
    },
    SCORING_SECTION_KEY: {
        PRE_NOTES_KEY: r"""PDDL calls their equivalent section (:metric ...), but I renamed because it made more sense to me. 

        Any syntax elements that are defined (because at some point a game needed them) but are currently unused (in the interactive games) will appear in {{ \color{{{unused_color}}} {unused_color} }}.
        """.format(unused_color=UNUSED_RULE_COLOR)
    },
    PREDICATES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The predicates are not defined as part of the DSL, but rather I envision them is being specific to a domain and being specified to any model as an input or something to be conditioned on. \\
            
            The following describes all predicates currently found in the interactive experiment games. Any predicates I forgot to provide a description for will appear in {{ \color{{{undescribed_color}}} {undescribed_color} }}.
        """.format(undescribed_color=UNDESCRIBED_ELEMENT_COLOR)
    },
    TYPES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The types are also not defined as part of the DSL, but I envision them as operating similarly to the predicates. \\
            
            The following describes all types currently found in the interactive experiment games. Any types I forgot to provide a description for will appear in {{\color{{{undescribed_color}}}{undescribed_color} }}.
        """.format(undescribed_color=UNDESCRIBED_ELEMENT_COLOR)
    }
}

class DSLTranslator:
    def __init__(self, section_key, section_type, section_name=None, notes=NOTES):
        self.section_key = section_key
        self.section_type = section_type
        self.section_name = section_name if section_name is not None else section_key.capitalize()
        self.notes = notes

        self.lines = []

    def process(self, section_data):
        raise NotImplementedError()

    def output(self):
        section_notes = self.notes[self.section_key] if self.section_key in self.notes else {}
        return [
            f'\\subsection{{{self.section_name}}}',
            section_notes[PRE_NOTES_KEY] if PRE_NOTES_KEY in section_notes else '',
            f'\\begin{{{self.section_type}}}'
        ] + self.lines + [
            f'\\end{{{self.section_type}}}',
            section_notes[POST_NOTES_KEY] if POST_NOTES_KEY in section_notes else '',
        ]


class SectionTranslator(DSLTranslator):
    def __init__(self, section_key, core_blocks, additional_blocks=None, section_name=None, consider_used_rules=None,
                 section_type='grammar', unused_rule_color=UNUSED_RULE_COLOR):
        super().__init__(section_key, section_type, section_name)
        self.section_key = section_key
        self.core_blocks = core_blocks
        self.additional_blocks = additional_blocks if additional_blocks is not None else []
        self.consider_used_rules = consider_used_rules if consider_used_rules is not None else []
        self.unused_rule_color = unused_rule_color

        self.lines = []
        self.remaining_rules = []
        self.unused_rules = []     

    def process(self, section_data):
        keys = set(section_data.keys())

        for blocks, is_core in zip((self.core_blocks, self.additional_blocks), (True, False)):
            for block_text, block_rules in blocks:

                if isinstance(block_rules, str):
                    if block_rules not in keys and block_rules not in self.consider_used_rules:
                        if not is_core:
                            continue

                        block_text = f'{{ \\color{{{self.unused_rule_color}}} {block_text} }}'
                        self.unused_rules.append(block_rules)
                    elif block_rules in keys:
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

DEFAULT_RULE_SECTION_TYPE = 'lstlisting' 

class RuleTypeTranslator(DSLTranslator):
    def __init__(self, section_key, rule_to_value_extractors, output_lines_func, section_type=DEFAULT_RULE_SECTION_TYPE, descriptions=None, section_name=None):
        super().__init__(section_key, section_type, section_name)
        self.rule_to_value_extractors = rule_to_value_extractors
        self.output_lines_func = output_lines_func
        self.count_data = defaultdict(lambda: 0)
        self.additional_data = defaultdict(lambda: defaultdict(list))
        self.descriptions = descriptions if descriptions is not None else {}

    def process(self, section_data):
        for rule, (key_extractor, additional_extractors) in self.rule_to_value_extractors.items():
            if rule in section_data:
                for instance in section_data[rule]:
                    instance_key = instance[key_extractor] if isinstance(key_extractor, str) else key_extractor(instance)

                    if instance_key:
                        if isinstance(instance_key, str):
                            self._handle_single_key(instance, instance_key, additional_extractors)

                        else:
                            for key in instance_key:
                                self._handle_single_key(instance, key, additional_extractors)

    def _handle_single_key(self, instance, instance_key, additional_extractors):
        self.count_data[instance_key] += 1

        if additional_extractors:
            for additional_extractor in additional_extractors:
                key, value = additional_extractor(instance, instance_key)
                if key is not None:
                    self.additional_data[instance_key][key].append(value)

    def output(self):
        self.lines = self.output_lines_func(self.count_data, self.additional_data, self.descriptions)
        return super().output()


def predicate_data_to_lines(count_data, additional_data, descriptions):
    lines = []

    for key in sorted(count_data.keys()):
        count = count_data[key]
        n_args = set(additional_data[key]['n_args'])
        n_args = list(n_args)[0] if len(n_args) == 1 else None
        arg_str = ' '.join([f'<arg{i + 1}>' for i in range(n_args)]) if n_args is not None else '<ambiguous arguments>'

        line = f'({key} {arg_str}) [{count} references] ; {descriptions[key] if key in descriptions else ""}'
        if key not in descriptions:
            line = f'(* \\color{{{UNDESCRIBED_ELEMENT_COLOR}}}) {line} *)'

        lines.append(line)

    return lines


def type_data_to_lines(count_data, additional_data, descriptions):
    lines = []

    for key in sorted(count_data.keys(), key=_type_name_sorting):
        count = count_data[key]
        # TODO: consider doing something with co-ocurrence data
        line = f'{key} [{count} references] ; {descriptions[key] if key in descriptions else ""}'
        if key not in descriptions:
            line = f'(* \\color{{{UNDESCRIBED_ELEMENT_COLOR}}} {line} *)'
        
        lines.append(line)

    return lines


def _type_name_sorting(type_name):
    if type_name == 'game_object': return -1
    if type_name == 'block': return 0
    if '_block' in type_name: return 1
    if type_name == 'ball': return 2
    if 'ball' in type_name: return 3
    return ord(type_name[0])


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

<predicate-term> ::= <name> | <variable> | <predicate> "#" In at least one case, I wanted to have a predicate act on other predicates, but that doesn't really make sense. See the discussion of the (side ...) predicate below.""",
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

<pref-forall> ::= (forall <variable-list> <preference>) "#" this syntax is used to specify variants of the same preference for different object, which differ in their scoring. These are specified using the <pref-name-and-types> syntax element's optional types, see scoring below.
    
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

    (r'<count-nonoverlapping> ::= (count-nonoverlapping <name>) "#" count how many times the preference is satisfied by non-overlapping sequences of states ', 'count_nonoverlapping'),
    (r'<count-once> ::= (count-once <name>) "#" count whether or not this preference was satisfied at all', 'count_once'),
    (r'<count-once-per-objects> ::= (count-once-per-objects <name>) "#" count once for each unique combination of objects quantified in the preference that satisfy it', 'count_once_per_objects'),
    (r'<count-longest> ::= (count-longest <name>) "#" count the longest (by number of states) satisfication of this preference', 'count_longest'),
    (r'<count-shortest> ::= (count-shortest <name>) "#" count the shortest satisfication of this preference ', 'count_shortest'),
    (r'<count-total> ::= (count-total <name>) "#" count how many states in total satisfy this preference', 'count_total'),
    (r'<count-increasing-measure> ::= (count-increasing-measure <name>) "#" currently unused, will clarify definition if it surfaces again', 'count_increasing_measure'),
    (r'<count-unique-positions> ::= (count-unique-positions <name>) "#" count how many times the preference was satisfied with quantified objects that remain stationary within each preference satisfcation, and have different positions between different satisfactions.', 'count_unique_positions'),

    (r"""<pref-name-and-types> ::= <name> <pref-object-type>$^*$ "#" the optional <pref-object-type>s are used to specify a particular variant of the preference for a given object, see the <pref-forall> syntax above.

    <pref-object-type> ::= : <name>
    """, ('pref_name_and_types', 'pref_object_type')),
)

PREDICATE_DESCRIPTIONS = {
    'above': 'is the first object above the second object?',
    'adjacent': 'are the two objects adjacent? [will probably be implemented as distance below some threshold]',
    'agent_crouches': 'is the agent crouching?',
    'agent_holds': 'is the agent holding the object?',
    'in': 'is the second argument inside the first argument? [a containment check of some sort, for balls in bins, for example]',
    'in_building': 'Is the object part of a building? (* \\textbf I dislike this predicate, which I previously used as a crutch, and I am trying to find alternatives around it *)',
    'in_motion': 'Is the object in motion?',
    'object_orientation': 'Is the first argument, an object, in the orientation specified by the second argument? Used to check if an object is upright or upside down',
    'on': 'Is the second object on the first one?',
    'open': 'Is the object open? Only valid for objects that can be opened, such as drawers.',
    'opposite': 'So far used only with walls, or sides of the room, to specify two walls opposite each other in conjunction with other predicates involving these walls',
    'side': '(* \\textbf This is not truly a predicate, and requires a more tight solution. I so far used it as a crutch to specify that two particular sides of objects are adjacent, for example (adjacent (side ?h front) (side ?c back)). But that makes (side <object> <side-def>) a function returning an object, not a predicate, where <side-def> is front, back, etc.. Maybe it should be something like (adjacent-side <object1> <side-def1> <object2> <side-def2>)? *)',
    'touch': 'Are these two objects touching?',
    'type': 'Is the first argument, an object, an instance of the type specified by the second argument?',    
}

TYPE_DESCRIPTIONS = {
    'game_object': 'Parent type of all objects',
    'block': 'Parent type of all block types:',
    'bridge_block': '.',
    'cube_block': '.',
    'flat_block': '.',
    'pyramid_block': '.',
    'tall_cylindrical_block': '.',
    'ball': 'Parent type of all ball types:',
    'dodgeball': '.',
    'golfball': '.',
    'chair': ',',
    'curved_wooden_ramp': '.',
    'doggie_bed': '.',
    'hexagonal_bin': '.',
    'large_triangular_ramp': '.',
    'pillow': '.',
    'teddy_bear': '.',
    'top_drawer': 'The top of the two drawers in the nightstand near the bed. (* \\textbf Do we want to specify this differently? *)',
    'wall': 'One of the walls in the room',
}


def extract_n_args(ast, key=None):
    n_args = 0
    if 'pred_args' in ast:
        if isinstance(ast.pred_args, str):
            n_args = 1
        else:
            n_args = len(ast.pred_args)

    return ('n_args', n_args)


PREDICATE_RULES = {
    PREDICATE: ('pred_name', (extract_n_args, ))
}


def extract_single_variable_type(ast):
    if 'var_type' in ast and isinstance(ast.var_type, str):
        return ast.var_type

    return None


def extract_either_variable_types(ast):
    if 'type_names' in ast:
        if isinstance(ast.type_names, str):
            return (ast.type_names, )

        return ast.type_names

    return None


def extract_co_ocurring_types(ast, key):
    if 'type_names' in ast:
        if not isinstance(ast.type_names, str):
            types = list(ast.type_names)
            if key in types: 
                types.remove(key)

            return ('co_ocurring_types', types)

    return (None, None)

TYPE_RULES = {
    'variable_type_def': (extract_single_variable_type, None),
    'either_types': (extract_either_variable_types, None)
}


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar) 

    asts = load_asts(args, grammar_parser, should_print=args.print_dsls)
    parser = DSLToLatexParser(args.template_file, args.output_file)

    setup_translator = SectionTranslator(SETUP_SECTION_KEY, SETUP_BLOCKS, (SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE]))
    pref_translator = SectionTranslator(PREFERENCES_SECTION_KEY, PREFERENCES_BLOCKS, (SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE]), section_name='Preferences')
    terminal_translator = SectionTranslator(TERMINAL_SECTION_KEY, TERMINAL_BLOCKS, None, section_name='Terminal Conditions')
    scoring_translator = SectionTranslator(SCORING_SECTION_KEY, SCORING_BLOCKS, None, consider_used_rules=['preference-eval',])

    predicate_translator = RuleTypeTranslator(PREDICATES_SECTION_KEY, PREDICATE_RULES, predicate_data_to_lines, descriptions=PREDICATE_DESCRIPTIONS)
    type_translator = RuleTypeTranslator(TYPES_SECTION_KEY, TYPE_RULES, type_data_to_lines, descriptions=TYPE_DESCRIPTIONS)

    # TODO: handle mathematical definitions and open question in code, rather than in template
    # TODO: compile latex to pdf immediately from Python?

    parser.register_processor(setup_translator)
    parser.register_processor(pref_translator)
    parser.register_processor(terminal_translator)
    parser.register_processor(scoring_translator)

    parser.register_processor(predicate_translator)
    parser.register_processor(type_translator)

    for ast in asts:
        parser(ast)

    parser.process()
    parser.output()

    out_dir, original_out_file = os.path.split(args.output_file)
    out_dir = os.path.join(out_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, original_out_file)
    shutil.copy(args.output_file, out_file)

    os.system(f'cd {out_dir} && pdflatex -shell-escape {original_out_file}')
    pdf_out_path = os.path.splitext(out_file)[0] + '.pdf'
    pdf_final_path = args.output_file.replace('.tex', '.pdf')
    shutil.copy(pdf_out_path, pdf_final_path)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
