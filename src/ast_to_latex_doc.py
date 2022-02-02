import argparse
from collections import defaultdict, namedtuple
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
parser.add_argument('-c', '--compile-pdf', action='store_true')
parser.add_argument('-n', '--new-data-start')
parser.add_argument('-e', '--external', action='store_true')
parser.add_argument('-u', '--omit-unused-rules', action='store_true')


DEFAULT_PREFIX_LINES = [
    r'\section{DSL Docuemntation as of \today}',
    '',
    # r'\subsection*{Color-coding}',
    # r'\begin{itemize}',
    # r'\item {\color{red} \textbf{Undefined terms (red)}}: a term that appears somewhere that I forgot to provide a definition for',
    # r'\item {\color{gray} \textbf{Unused terms (gray)}}: a term that appears in the definitions I documented but does not appear in any games',
    # r'\item {\color{teal} \textbf{New terms (teal)}}: a term that appears for the first time in the newest batch of games I translated',
    # r'\end{itemize}',
    # ''
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
    def __init__(self, template_path, output_path, new_data_start=None, prefix_lines=DEFAULT_PREFIX_LINES, postfix_lines=None) -> None:
        super().__init__()
        self.template_path = template_path
        self.output_path = output_path
        self.new_data_start = new_data_start
        self.data_is_new = False

        self.rules_by_section = defaultdict(lambda: defaultdict(list))
        self.processors = []
        self.prefix_lines = prefix_lines if prefix_lines is not None else []
        self.postfix_lines = postfix_lines if postfix_lines is not None else []

    def register_processor(self, processor):
        self.processors.append(processor)

    def __call__(self, ast, **kwargs):
        if isinstance(ast, tuple) and ast[0] == '(define' and \
            self.new_data_start is not None and ast[1].game_name == self.new_data_start:

            self.data_is_new = True
        
        return super().__call__(ast, **kwargs)

    def _handle_tuple(self, ast, **kwargs):
        if len(ast) > 0 and isinstance(ast[0], str) and ast[0].startswith('(:'):
            kwargs['section'] = ast[0][2:]

        return super()._handle_tuple(ast, **kwargs)

    def _handle_ast(self, ast, **kwargs):
        if 'section' in kwargs and ast.parseinfo:
            self.rules_by_section[kwargs['section']][ast.parseinfo.rule].append((ast, self.data_is_new))

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


UNUSED_RULE_OR_ELEMENT_COLOR = 'gray'
NEW_RULE_OR_ELEMENT_COLOR = 'teal'
UNDESCRIBED_ELEMENT_COLOR = 'red'

SETUP_SECTION_KEY = 'setup'
PREFERENCES_SECTION_KEY = 'constraints'
TERMINAL_SECTION_KEY = 'terminal'
SCORING_SECTION_KEY = 'scoring'
PREDICATES_SECTION_KEY = 'predicates'
FUNCTIONS_SECTION_KEY = 'functions'
TYPES_SECTION_KEY = 'types'

PRE_NOTES_KEY = 'pre'
POST_NOTES_KEY = 'post'

NOTES = {
    SETUP_SECTION_KEY: {
        PRE_NOTES_KEY: r"""PDDL doesn't have any close equivalent of this, but when reading through the games participants specify, 
        they often require some transformation of the room from its initial state before the game can be played.
        We could treat both as parts of the gameplay, but we thought there's quite a bit to be gained by splitting them -- for example,
        the policies to setup a room are quite different from the policies to play a game (much more static). \\

        The one nuance here came from the (game-conserved ...) and (game-optional ...) elements. It seemed to us that some setup elements should be maintained
        throughout gameplay (for example, if you place a bin somewhere to throw into, it shouldn't move unless specified otherwise).
        Other setup elements can, or often must change -- for example, if you set the balls on the desk to throw them, you'll have to pick them up off the desk to throw them.
        These elements provide that context, which could be useful for verifying that agents playing the game don't violate these conditions.
        """,
    },
    PREFERENCES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The gameplay preferences specify the core of a game's semantics, capturing how a game should be played by specifying temporal constraints over predicates. 
        
        PDDL calls their temporal preferences 'constraints', but that's not entirely the right name for us. Maybe we should rename? \\
        """.format(unused_color=UNUSED_RULE_OR_ELEMENT_COLOR) # Any syntax elements that are defined (because at some point a game needed them) but are currently unused (in the interactive games) will appear in {{ \color{{{unused_color}}} {unused_color} }}.
    },
    TERMINAL_SECTION_KEY: {
        PRE_NOTES_KEY: r"""Some participants explicitly specify terminal conditions, but we consider this to be optional. 
        """,
        POST_NOTES_KEY: r"""For a full specification of the \textlangle scoring-expr\textrangle\ token, see the scoring section below.
        """
    },
    SCORING_SECTION_KEY: {
        PRE_NOTES_KEY: r"""Scoring rules specify how to count preferences (count once, once for each unique objects that fulfill the preference, each time a preference is satisfied, etc.), and the arithmetic to combine
        counted preference statisfactions to get a final score.
        
        PDDL calls their equivalent section (:metric ...), but we renamed because it made more sense to in the context of games. 
        """.format(unused_color=UNUSED_RULE_OR_ELEMENT_COLOR) # Any syntax elements that are defined (because at some point a game needed them) but are currently unused (in the interactive games) will appear in {{ \color{{{unused_color}}} {unused_color} }}.
    },
    PREDICATES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The predicates are not defined as part of the DSL, but rather we envision them is being specific to a domain and being specified to any model as an input or something to be conditioned on. \\
            
            The following describes all predicates currently found in our game dataset:
        """.format(undescribed_color=UNDESCRIBED_ELEMENT_COLOR) # Any predicates I forgot to provide a description for will appear in {{ \color{{{undescribed_color}}} {undescribed_color} }}.
    },
    FUNCTIONS_SECTION_KEY: {
        PRE_NOTES_KEY: r"""Functions operate simlarly to predicates, but rather than returning a boolean value, they return a numeric value or a type. 
        Similarly to predicates, they are not parts of the DSL per se, but might vary by environment.

        The following describes all functions currently found in our game dataset:
        """,
    },
    TYPES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The types are also not defined as part of the DSL, but we envision them as operating similarly to the predicates. \\
            
            The following describes all types currently found in our game dataset: 
        """.format(undescribed_color=UNDESCRIBED_ELEMENT_COLOR) # Any types we forgot to provide a description for will appear in {{\color{{{undescribed_color}}}{undescribed_color} }}.
    }
}

class DSLTranslator:
    def __init__(self, section_key, section_type, section_name=None, external_mode=False, omit_unused=False, notes=NOTES):
        self.section_key = section_key
        self.section_type = section_type
        self.section_name = section_name if section_name is not None else section_key.capitalize()

        self.external_mode = external_mode
        self.omit_unused = omit_unused

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
                 section_type='grammar', unused_rule_color=UNUSED_RULE_OR_ELEMENT_COLOR, new_rule_color=NEW_RULE_OR_ELEMENT_COLOR,
                 external_mode=False, omit_unused=False):
        super().__init__(section_key, section_type, section_name, external_mode=external_mode, omit_unused=omit_unused)
        self.section_key = section_key
        self.core_blocks = core_blocks
        self.additional_blocks = additional_blocks if additional_blocks is not None else []
        self.consider_used_rules = consider_used_rules if consider_used_rules is not None else []
        self.unused_rule_color = unused_rule_color
        self.new_rule_color = new_rule_color

        self.lines = []
        self.remaining_rules = []
        self.unused_rules = []     

    def process(self, section_data):
        keys = set(section_data.keys())

        for blocks, is_core in zip((self.core_blocks, self.additional_blocks), (True, False)):
            for block_text, block_rules in blocks:

                if isinstance(block_rules, str):
                    if block_rules not in keys and block_rules not in self.consider_used_rules:
                        self.unused_rules.append(block_rules)
                        
                        if not is_core or self.omit_unused:
                            continue

                        if not self.external_mode:
                            block_text = f'{{ \\color{{{self.unused_rule_color}}} {block_text} }}'
                        
                    elif block_rules in keys:
                        _, block_is_new = zip(*section_data[block_rules])
                        rule_is_new = all(block_is_new)
                        if rule_is_new and not self.external_mode:
                            block_text = f'{{ \\color{{{self.new_rule_color}}} {block_text} }}'

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
    def __init__(self, section_key, rule_to_value_extractors, output_lines_func, section_type=DEFAULT_RULE_SECTION_TYPE, 
        descriptions=None, section_name=None, external_mode=False, omit_unused=False):

        super().__init__(section_key, section_type, section_name, external_mode=external_mode, omit_unused=omit_unused)
        self.rule_to_value_extractors = rule_to_value_extractors
        self.output_lines_func = output_lines_func
        self.count_data = defaultdict(lambda: 0)
        self.is_new_data = defaultdict(lambda: True)
        self.additional_data = defaultdict(lambda: defaultdict(list))
        self.descriptions = descriptions if descriptions is not None else {}

    def process(self, section_data):
        for rule, (key_extractor, additional_extractors) in self.rule_to_value_extractors.items():
            if rule in section_data:
                for instance, instance_is_new in section_data[rule]:
                    instance_key = instance[key_extractor] if isinstance(key_extractor, str) else key_extractor(instance)

                    if instance_key:
                        if isinstance(instance_key, str):
                            self._handle_single_key(instance, instance_key, instance_is_new, additional_extractors)

                        else:
                            for key in instance_key:
                                self._handle_single_key(instance, key, instance_is_new, additional_extractors)

    def _handle_single_key(self, instance, instance_key, instance_is_new, additional_extractors):
        self.count_data[instance_key] += 1
        self.is_new_data[instance_key] = self.is_new_data[instance_key] and instance_is_new

        if additional_extractors:
            for additional_extractor in additional_extractors:
                key, value = additional_extractor(instance, instance_key)
                if key is not None:
                    self.additional_data[instance_key][key].append(value)

    def output(self):
        self.lines = self.output_lines_func(self.count_data, self.is_new_data, self.additional_data, self.descriptions, self.external_mode)
        return super().output()


def _format_line_to_color(line, color):
    return f'(*\\color{{{color}}} {line}*)'


def predicate_data_to_lines(count_data, is_new_data, additional_data, descriptions, external_mode=False, omit_unused=False):
    lines = []

    for key in sorted(count_data.keys()):
        count = count_data[key]
        n_args = list(sorted(set(additional_data[key]['n_args'])))
        arg_str = ' '.join([f'<arg{i + 1}>' for i in range(n_args[0])]) if len(n_args) == 1 else f'<{" or ".join([str(n) for n in n_args])} arguments>'

        line = f'({key} {arg_str}) [{count} reference{"s" if count != 1 else ""}] {"; " + descriptions[key] if key in descriptions else ""}'
        if key not in descriptions and not external_mode:
            line = _format_line_to_color(line, UNDESCRIBED_ELEMENT_COLOR)

        elif is_new_data[key] and not external_mode:
            line = _format_line_to_color(line, NEW_RULE_OR_ELEMENT_COLOR)

        lines.append(line)

    return lines


TypeDesc = namedtuple('TypeDesc', ('key', 'description', 'preformatted'), defaults=(None, '', False))


def section_separator_typedesc(section_name):
    return TypeDesc(section_name, f'---------- (* \\textbf{{{section_name}}} *) ----------', True)


def type_data_to_lines(count_data, is_new_data, additional_data, descriptions, external_mode=False, omit_unused=False):
    lines = []
    unused_keys = set(count_data.keys())

    for type_desc in descriptions:
        key, description, preformatted = type_desc

        if preformatted:
            lines.append(description)

        else:
            if key in unused_keys: 
                unused_keys.remove(key)

            count = count_data[key] if key in count_data else 0
            # TODO: consider doing something with co-ocurrence data
            line = f'{key} [{count if count != 0 else "N/A"} reference{"s" if count != 1 else ""}] {"; " + description if description else ""}'

            if count == 0:
                if omit_unused:
                    continue

                if not external_mode:
                    line = _format_line_to_color(line, UNUSED_RULE_OR_ELEMENT_COLOR)

            elif is_new_data[key] and not external_mode:
                line = _format_line_to_color(line, NEW_RULE_OR_ELEMENT_COLOR)

            lines.append(line)

    if unused_keys and not external_mode:
        lines.append(section_separator_typedesc('Undescribed types').description)
        for key in unused_keys:
            count = count_data[key]
            line = _format_line_to_color(f'{key} [{count} reference{"s" if count != 1 else ""}]', UNDESCRIBED_ELEMENT_COLOR)
            lines.append(line)

    return lines


# def _type_name_sorting(type_name):
#     if type_name == 'game_object': return -1
#     if type_name == 'block': return 0
#     if '_block' in type_name: return 1
#     if type_name == 'ball': return 2
#     if 'ball' in type_name: return 3
#     return ord(type_name[0])


FUNCTION_COMPARISON = 'function_comparison'
VARIABLE_LIST = 'variable_list'
PREDICATE = 'predicate'
FUNCTION = 'function_eval'
SHARED_BLOCKS = {
    FUNCTION_COMPARISON: (r"""<f-comp> ::= (<comp-op> <function-eval-or-number> <function-eval-or-number>) \alt
    (= <function-eval-or-number>$^+$)
    
<comp-op> ::=  \textlangle \ | \textlangle = \ | = \ | \textrangle \ | \textrangle =

<function-eval-or-number> ::= <function-eval> | <number>

<function-eval> ::= (<name> <function-term>$^+$)

<function-term> ::= <name> | <variable> | <number> | <predicate>""", ('function_comparison', FUNCTION)),

    VARIABLE_LIST: (r"""<variable-list> ::= (<variable-type-def>$^+$)

<variable-type-def> ::= <variable>$^+$ - <type-def>

<variable> ::= /\textbackslash?[a-z][a-z0-9]*/  "#" a question mark followed by a letter, optionally followed by additional letters or numbers

<type-def> ::= <name> | <either-types>

<either-types> ::= (either <name>$^+$)""", ('variable_list', 'variable_type_def', 'either_types')),

    PREDICATE: (r"""<predicate> ::= (<name> <predicate-term>$^*$)

<predicate-term> ::= <name> | <variable>""",
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
    (exists (<typed list(variable)>) <setup-predicate>) \alt
    (forall (<typed list(variable)>) <setup-predicate>) \alt
    <f-comp> \alt
    <predicate>""", ('setup_and_predicate', 'setup_or_predicate', 'setup_not_predicate', 'setup_forall_predicate', 'setup_exists_predicate')),
)


PREFERENCES_BLOCKS = (
    (r"""<constraints> ::= <pref-def> | (and <pref-def>$^+$)
    
<pref-def> ::= <pref-forall> | <preference> 

<pref-forall> ::= (forall <variable-list> <preference>) "#" this syntax is used to specify variants of the same preference for different object, which differ in their scoring. These are specified using the <pref-name-and-types> syntax element's optional types, see scoring below.
    
<preference> ::= (preference <name> <preference-quantifier>)

<preference-quantifier> ::= (exists (<variable-list>) <preference-body> 
\alt  (forall (<variable-list>) <preference-body>)
\alt <preference-body>) 

<preference-body> ::=  <then> | <at-end> | <always> """, ('preference', 'pref_forall', 'pref_body_exists')),

    (r'<at-end> ::= (at-end <pref-predicate>)', 'at_end'), 

    # (r'<always> ::= (always <pref-predicate>)', 'always'), 

    (r"""<then> ::= (then <seq-func> <seq-func>$^+$) 

<seq-func> ::= <once> | <once-measure> | <hold> | <hold-while>""", 'then'),  #  | <hold-for> | <hold-to-end> \alt <forall-seq>

    (r'<once> ::= (once <pref-predicate>) "#" The predicate specified must hold for a single world state', 'once'),

    (r'<once-measure> ::= (once <pref-predicate> <function-eval>) "#" The predicate specified must hold for a single world state, and record the value of the function evaluation', 'once_measure'),

    (r'<hold> ::= (hold <pref-predicate>) "#" The predicate specified must hold for every state between the previous temporal operator and the next one', 'hold'),

    (r'<hold-while> ::= (hold-while <pref-predicate> <pref-predicate>$^+$) "#" The predicate specified must hold for every state between the previous temporal operator and the next one. While it does, at least one state must satisfy each of the predicates specified in the second arumgnet onward'  , 'while_hold'),

    # (r'<hold-for> ::= (hold-for <number> <pref-predicate>)', 'hold_for'),

    # (r'<hold-to-end> ::= (hold-to-end <pref-predicate>)', 'hold_to_end'),

    # (r'<forall-seq> ::= (forall-sequence (<variable-list>) <then>)', 'forall_seq'),

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

<pref-predicate-forall> ::= (forall <variable-list> <pref-predicate>)""", ('pref_predicate_and', 'pref_predicate_or', 'pref_predicate_not', 'pref_predicate_exists', 'pref_predicate_forall' ))
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
        
    """, ('scoring_multi_expr', 'scoring_binary_expr', 'scoring_neg_expr')),

    (r"""<scoring-comp> ::=  (<comp-op> <scoring-expr> <scoring-expr>) \alt
        (= <scoring-expr>$^+$)
    """, 'scoring_comp'),

    (r"""<preference-eval> ::=  <count-nonoverlapping> \alt
        <count-once> \alt
        <count-once-per-objects> \alt
        <count-nonoverlapping-measure> \alt
        <count-unique-positions> \alt
        <count-same-positions> \alt
        <count-maximal-nonoverlapping> \alt
        <count-maximal-overlapping> \alt
        <count-maximal-once-per-objects> \alt
        <count-maximal-once> \alt        
        <count-once-per-external-objects> 

    """, 'preference-eval'),

    (r'<count-nonoverlapping> ::= (count-nonoverlapping <pref-name-and-types>) "#" count how many times the preference is satisfied by non-overlapping sequences of states ', 'count_nonoverlapping'),
    (r'<count-once> ::= (count-once <pref-name-and-types>) "#" count whether or not this preference was satisfied at all', 'count_once'),
    (r'<count-once-per-objects> ::= (count-once-per-objects <pref-name-and-types>) "#" count once for each unique combination of objects quantified in the preference that satisfy it', 'count_once_per_objects'),
    # (r'<count-longest> ::= (count-longest <pref-name-and-types>) "#" count the longest (by number of states) satisfication of this preference', 'count_longest'),
    # (r'<count-shortest> ::= (count-shortest <pref-name-and-types>) "#" count the shortest satisfication of this preference ', 'count_shortest'),
    # (r'<count-total> ::= (count-total <pref-name-and-types>) "#" count how many states in total satisfy this preference', 'count_total'),
    # (r'<count-increasing-measure> ::= (count-increasing-measure <pref-name-and-types>) "#" currently unused, will clarify definition if it surfaces again', 'count_increasing_measure'),
    (r'<count-nonoverlapping-measure> ::= (count-nonoverlapping-measure <pref-name-and-types>) "#" Can only be used in preferences including a <once-measure> modal, maps each preference satistifaction to the value of the function evaluation in the <once-measure>', 'count_nonoverlapping_measure'),
    (r'<count-unique-positions> ::= (count-unique-positions <pref-name-and-types>) "#" count how many times the preference was satisfied with quantified objects that remain stationary within each preference satisfcation, and have different positions between different satisfactions.', 'count_unique_positions'),
    (r'<count-same-positions> ::= (count-same-positions <pref-name-and-types>) "#" count how many times the preference was satisfied with quantified objects that remain stationary within each preference satisfcation, and have (approximately) the same position between different satisfactions.', 'count_same_positions'),
    (r'<note> : "#" All of the count-maximal-... operators refer to counting only for preferences inside a (forall ...), and count only for the object quantified externally that has the most preference satisfactions to it. If there exist multiple preferences in a single (forall ...) block, score for the single object that satisfies the most over all such preferences.', 'maximal_explainer'),
    (r'<count-maximal-nonoverlapping> ::= (count-maximal-nonoverlapping <pref-name-and-types>) "#" For the single externally quantified object with the most satisfcations, count non-overlapping satisfactions of this preference', 'count_maximal_nonoverlapping'),
    (r'<count-maximal-overlapping> ::= (count-maximal-overlapping <pref-name-and-types>) "#" For the single externally quantified object with the most satisfcations, count how many satisfactions of this preference with different objects overlap in their states', 'count_maximal_overlapping'),
    (r'<count-maximal-once-per-objects> ::= (count-maximal-once-per-objects <pref-name-and-types>) "#" For the single externally quantified object with the most satisfcations, count this preference for each set of quantified objects that satisfies it', 'count_maximal_once_per_objects'),
    (r'<count-maximal-once> ::= (count-maximal-once <pref-name-and-types>) "#" For the externally quantified object with the most satisfcations (across all preferences in the same (forall ...) block), count this preference at most once', 'count_maximal_once'),
    (r'<count-once-per-external-objects> ::=  (count-once-per-external-objects <pref-name-and-types>) "#" Similarly to count-once-per-objects, but counting only for each unique object or combination of objects quantified in the (forall ...) block including this preference',  'count_once_per_external_objects'),

    (r"""<pref-name-and-types> ::= <name> <pref-object-type>$^*$ "#" the optional <pref-object-type>s are used to specify a particular variant of the preference for a given object, see the <pref-forall> syntax above.

    <pref-object-type> ::= : <name>
    """, ('pref_name_and_types', 'pref_object_type')),
)

PREDICATE_DESCRIPTIONS = {
    '=': 'Are these two objects the same object?',
    'above': 'Is the first object above the second object?',
    'adjacent': 'Are the two objects adjacent? [will probably be implemented as distance below some threshold]',
    'adjacent_side': 'Are the two objects adjacent on the sides specified? Specifying a side for the second object is optional, allowing to specify <obj1> <side1> <obj2> or <obj1> <side1> <obj2> <side2>',
    'agent_crouches': 'Is the agent crouching?',
    'agent_holds': 'Is the agent holding the object?',
    'between': 'Is the second object between the first object and the third object?',
    'broken': 'Is the object broken?',
    'equal_x_position': 'Are these two objects (approximately) in the same x position? (in our environment, x, z are spatial coordinates, y is the height)',
    'equal_z_position': 'Are these two objects (approximately) in the same z position? (in our environment, x, z are spatial coordinates, y is the height)',
    'faces': 'Is the front of the first object facing the front of the second object?',
    'game_over': 'Is this the last state of gameplay?',
    'game_start': 'Is this the first state of gameplay?',
    'in': 'Is the second argument inside the first argument? [a containment check of some sort, for balls in bins, for example]',
    'in_motion': 'Is the object in motion?',
    'is_setup_object': 'Is this the object of the same type referenced in the setup?',
    'object_orientation': 'Is the first argument, an object, in the orientation specified by the second argument? Used to check if an object is upright or upside down',
    'on': 'Is the second object on the first one?',
    'open': 'Is the object open? Only valid for objects that can be opened, such as drawers.',
    'opposite': 'So far used only with walls, or sides of the room, to specify two walls opposite each other in conjunction with other predicates involving these walls',
    'rug_color_under': 'Is the color of the rug under the object (first argument) the color specified by the second argument?',
    'same_type': 'Are these two objects of the same type?',
    # 'side': '(* \\textbf This is not truly a predicate, and requires a more tight solution. I so far used it as a crutch to specify that two particular sides of objects are adjacent, for example (adjacent (side ?h front) (side ?c back)). But that makes (side <object> <side-def>) a function returning an object, not a predicate, where <side-def> is front, back, etc.. Maybe it should be something like (adjacent-side <object1> <side-def1> <object2> <side-def2>)? *)',
    'toggled_on': 'Is this object toggled on?',
    'touch': 'Are these two objects touching?',
    'type': 'Is the first argument, an object, an instance of the type specified by the second argument?',    
}

FUNCTION_DESCRIPTIONS = {
    'building_size': 'Takes in an argument of type building, and returns how many objects comprise the building (as an integer)',
    'color': 'Take in an argument of type object, and returns the color of the object (as a color type object)',
    'distance': 'Takes in two arguments of type object, and returns the distance between the two objects (as a floating point number)',
    'distance_side': 'Similarly to the adjacent_side predicate, but applied to distance. Takes in three or four arguments, either <obj1> <side1> <obj2> or <obj1> <side1> <obj2> <side2>, and returns the distance between the first object on the side specified to the second object (optionally to its specified side)',
    'type': 'Takes in an argument of type object, and returns the type of the object (as a string)',
    'x_position': 'Takes in an argument of type object, and returns the x position of the object (as a floating point number)',
}


TYPE_DESCRIPTIONS = (
    TypeDesc('game_object', 'Parent type of all objects'),
    TypeDesc('agent', 'The agent'),
    TypeDesc('building', 'Not a real game object, but rather, a way to refer to structures the agent builds'),
    section_separator_typedesc('Blocks'),
	TypeDesc('block', 'Parent type of all block types:'),
	TypeDesc('bridge_block'),
	TypeDesc('cube_block'),
    TypeDesc('blue_cube_block'),
    TypeDesc('tan_cube_block'),
    TypeDesc('yellow_cube_block'),
	TypeDesc('flat_block'),
	TypeDesc('pyramid_block'),
    TypeDesc('blue_pyramid_block'),
    TypeDesc('red_pyramid_block'),
    TypeDesc('triangle_block'),
    TypeDesc('yellow_pyramid_block'),
	TypeDesc('cylindrical_block'),
	TypeDesc('tall_cylindrical_block'),
    section_separator_typedesc('Balls'),
	TypeDesc('ball', 'Parent type of all ball types:'),
	TypeDesc('beachball'),
	TypeDesc('basketball'),
	TypeDesc('dodgeball'),
    TypeDesc('blue_dodgeball'),
    TypeDesc('red_dodgeball'), #, '(* \\textbf Do we want to specify colored objects or not? *)'),
	TypeDesc('pink_dodgeball'), #, '(* \\textbf Do we want to specify colored objects or not? *)'),
	TypeDesc('golfball'),
    TypeDesc('green_golfball'), # '(* \\textbf Do we want to specify colored objects or not? *)'),
	section_separator_typedesc('Colors'),
    TypeDesc('color', 'Likewise, not a real game object, mostly used to refer to the color of the rug under an object'),
	TypeDesc('blue'),
    TypeDesc('brown'),
    TypeDesc('green'),
	TypeDesc('pink'),
    TypeDesc('orange'),
	TypeDesc('purple'),
    TypeDesc('red'),
    TypeDesc('tan'),
	TypeDesc('white'),
	TypeDesc('yellow'),
    section_separator_typedesc('Other moveable/interactable objects'),
	TypeDesc('alarm_clock'),
	TypeDesc('book'),
	TypeDesc('blinds', 'The blinds on the windows'),
	TypeDesc('chair'),
	TypeDesc('cellphone'),
	TypeDesc('cd'),
	TypeDesc('credit_card'),
	TypeDesc('curved_wooden_ramp'),
	TypeDesc('desktop'),
	TypeDesc('doggie_bed'),
	TypeDesc('hexagonal_bin'),
	TypeDesc('key_chain'),
	TypeDesc('lamp'),
	TypeDesc('laptop'),
    TypeDesc('main_light_switch', 'The main light switch on the wall'),
	TypeDesc('mug'),
	TypeDesc('triangular_ramp'),
	TypeDesc('green_triangular_ramp'), # '(* \\textbf Do we want to specify colored objects or not? *)'),
	TypeDesc('pen'),
    TypeDesc('pencil'),
    TypeDesc('pillow'),
	TypeDesc('teddy_bear'),
	TypeDesc('watch'),
    section_separator_typedesc('Immoveable objects'),
	TypeDesc('bed'),
    TypeDesc('corner', 'Any of the corners of the room'),
    TypeDesc('south_west_corner', 'The corner of the room where the south and west walls meet'),
    TypeDesc('door', 'The door out of the room'),
	TypeDesc('desk'),
    TypeDesc('desk_shelf', 'The shelves under the desk'),
	TypeDesc('drawer', 'Either drawer in the side table'),
	TypeDesc('top_drawer', 'The top of the two drawers in the nightstand near the bed.'), # (* \\textbf Do we want to specify this differently? *)'),
	TypeDesc('floor'),
    TypeDesc('rug'),
	TypeDesc('shelf'),
    TypeDesc('bottom_shelf'),
    TypeDesc('top_shelf'),
	TypeDesc('side_table', 'The side table/nightstand next to the bed'),
	TypeDesc('sliding_door', 'The sliding doors on the south wall (big windows)'),
    TypeDesc('east_sliding_door', 'The eastern of the two sliding doors (the one closer to the desk)'),
	TypeDesc('wall', 'Any of the walls in the room'),
    TypeDesc('north_wall', 'The wall with the door to the room'),
    TypeDesc('south_wall', 'The wall with the sliding doors'),
    TypeDesc('west_wall', 'The wall the bed is aligned to'),
    section_separator_typedesc('Non-object-type predicate arguments'),
    TypeDesc('back'),
    TypeDesc('front'),
    TypeDesc('left'),
    TypeDesc('right'),
    TypeDesc('sideways'),
    TypeDesc('upright'),
    TypeDesc('upside_down'),
    TypeDesc('front_left_corner', 'The front-left corner of a specific object (as determined by its front)'),
)


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

FUNCTION_RULES = {
    FUNCTION: ('func_name', (extract_n_args, ))
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


def extract_pref_name_and_types(ast):
    if 'object_types' in ast:
        if isinstance(ast.object_types, tatsu.ast.AST):
            return ast.object_types.type_name
        else:
            return [t.type_name for t in ast.object_types if 'type_name' in t]

    return None


def extract_types_from_predicates(ast):
    if 'pred_args' in ast:
        if isinstance(ast.pred_args, str) and not ast.pred_args.startswith('?'):
            return ast.pred_args

        filtered_args = [arg for arg in ast.pred_args if isinstance(arg, str) and not arg.startswith('?')]
        if filtered_args:
            return filtered_args

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
    'either_types': (extract_either_variable_types, None),
    'pref_name_and_types': (extract_pref_name_and_types, None),
    'predicate': (extract_types_from_predicates, None),
}

SCORING_CONSIDER_USED_RULES = (
    'scoring_maximize', 'scoring_minimize',
    'preference-eval', 'maximal_explainer', 'count_maximal_nonoverlapping', 
    'count_once_per_external_objects', 'scoring_neg_expr', 'pref_object_type', 
    'pref_name_and_types', 'count_nonoverlapping', 'count_once_per_objects',
    'count_once', 'count_maximal_nonoverlapping',
)


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar) 

    asts = load_asts(args, grammar_parser, should_print=args.print_dsls)
    parser = DSLToLatexParser(args.template_file, args.output_file, args.new_data_start)

    setup_translator = SectionTranslator(SETUP_SECTION_KEY, SETUP_BLOCKS, (SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE]), consider_used_rules=['setup_not', 'setup_statement'])
    pref_translator = SectionTranslator(PREFERENCES_SECTION_KEY, PREFERENCES_BLOCKS, (SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE]), section_name='Gameplay Preferences')
    terminal_translator = SectionTranslator(TERMINAL_SECTION_KEY, TERMINAL_BLOCKS, None, section_name='Terminal Conditions', consider_used_rules=['terminal_not'])
    scoring_translator = SectionTranslator(SCORING_SECTION_KEY, SCORING_BLOCKS, None, consider_used_rules=SCORING_CONSIDER_USED_RULES)

    predicate_translator = RuleTypeTranslator(PREDICATES_SECTION_KEY, PREDICATE_RULES, predicate_data_to_lines, descriptions=PREDICATE_DESCRIPTIONS)
    function_translator = RuleTypeTranslator(FUNCTIONS_SECTION_KEY, FUNCTION_RULES, predicate_data_to_lines, descriptions=FUNCTION_DESCRIPTIONS)
    type_translator = RuleTypeTranslator(TYPES_SECTION_KEY, TYPE_RULES, type_data_to_lines, descriptions=TYPE_DESCRIPTIONS)

    # TODO: handle mathematical definitions and open question in code, rather than in template

    parser.register_processor(setup_translator)
    parser.register_processor(pref_translator)
    parser.register_processor(terminal_translator)
    parser.register_processor(scoring_translator)

    parser.register_processor(predicate_translator)
    parser.register_processor(function_translator)
    parser.register_processor(type_translator)

    for ast in asts:
        parser(ast)

    parser.process()
    parser.output()

    if args.compile_pdf:
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
