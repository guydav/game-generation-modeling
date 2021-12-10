import argparse
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import os


parser = argparse.ArgumentParser()
DEFAULT_SCHEMA_FILE = './schema/game_schema.json'
parser.add_argument('-s', '--schema-file', default=DEFAULT_SCHEMA_FILE)
DEFAULT_TEST_FILES = (
    './schema/interactive_beta.json',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
# DEFAULT_OUTPUT_FILE = './latex/dsl_doc.tex'
# parser.add_argument('-o', '--output-file', default=DEFAULT_OUTPUT_FILE)
# parser.add_argument('-p', '--print-dsls', action='store_true')
# DEFAULT_TEMPLATE_FILE = './latex/template.tex'
# parser.add_argument('-f', '--template-file', default=DEFAULT_TEMPLATE_FILE)
# parser.add_argument('-c', '--compile-pdf', action='store_true')


def main(args):
    schema = json.load(open(args.schema_file))

    for test_path in args.test_files:
        with open(test_path) as test_file:
            test_file_json = json.load(test_file)
            games = test_file_json['games']

            for game in games:
                try:
                    # print(game)
                    validate(game, schema)

                except ValidationError as err:
                    print(f'In game {game["prolific_id"]} ({game["id"]}): {err.message}')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
