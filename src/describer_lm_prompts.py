###
# Prompts used for getting GPT-X to generate natural language (ish) descriptions of games. Prompts are broken down according to which 'stage' they map from:
# -Stage 0: raw game code
# -Stage 1: templated game description
# -Stage 2: natural language conversion of template
# -Stage 3: natural language description of game
###

SETUP_STAGE_1_TO_STAGE_2_PROMPT = """Your task is to convert a templated description of a game's setup into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:

### TEMPLATED DESCRIPTION:
In order to set up the game, the following must all be true for at least one time step:
- for any object ?d of type dodgeball or cube_block, it's not the case that there exists an object ?s of type shelf, such that ?d is on ?s
- main_light_switch is toggled on
- desktop is toggled on

### NATURAL LANGUAGE DESCRIPTION:
In order to set up this game, take every dodgeball and cube block off of the shelves. Then, make sure that the light switch and desktop are both turned on.

### TEMPLATED DESCRIPTION:
In order to set up the game, there exists an object ?h of type hexagonal_bin and an object ?b of type building, such that the following must all be true for at least one time step:
- ?b is on ?h
- for any object ?c of type cube_block, ?c is inside of ?b
- there exists an object ?c1 of type cube_block, an object ?c2 of type cube_block, an object ?c3 of type cube_block, an object ?c4 of type cube_block, an object ?c5 of type cube_block, and an object ?c6 of type cube_block, such that (?c1 is on ?h), (?c2 is on ?h), (?c3 is on ?h), (?c4 is on ?c1), (?c5 is on ?c2), and (?c6 is on ?c4)

and in addition, the following must all be true for every time step:
- ?h is adjacent to bed

### NATURAL LANGUAGE DESCRIPTION:
In order to set up the game, make a building out of every cube block and put it on the hexagonal bin. Three cube blocks need to be on the bin, two of them need to be on the first layer, and the last one needs to be on the second layer. 
Then, put the hexagonal bin next to the bed and make sure it stays there for the whole game.

Now, convert the following description:
### TEMPLATED DESCRIPTION:
{0}"""

CONSTRAINTS_STAGE_1_TO_STAGE_2_PROMPT = """Your task is to convert a templated description of a game's rules (expressed as "preferences") into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:

### TEMPLATED DESCRIPTION:
Preference 1: 'blockInTowerKnockedByDodgeball'
The variables required by this preference are:
- ?b: of type building
- ?c: of type cube_block
- ?d: of type dodgeball
- ?h: of type hexagonal_bin

This preference is satisfied when:
- first, there is a state where (the agent is holding ?d), (?b is on ?h), and (?c is inside of ?b)
- next, there is a sequence of one or more states where (it's not the case that the agent is holding ?d) and (?d is in motion) Additionally, during this sequence there is a state where ((?c touches ?d) or (there exists an object ?c2 of type cube_block, such that ?c2 touches ?c)) and a state where (?c is in motion) (in that order).
- finally, there is a state where it's not the case that ?c is in motion

Preference 2: 'throwAttempt'
The variables required by this preference are:
- ?d: of type dodgeball

This preference is satisfied when:
- first, there is a state where the agent is holding ?d
- next, there is a sequence of one or more states where (it's not the case that the agent is holding ?d) and (?d is in motion)
- finally, there is a state where it's not the case that ?d is in motion

### NATURAL LANGUAGE DESCRIPTION:
Preference 1: 'blockInTowerKnockedByDodgeball'
This preference is satisfied when:
- first, the agent picks up a dodgeball while there's a building on the hexagonal bin that has a cube in it
- next, the agent throws the dodgeball at the building, causing the dodgeball to hit the cube and the cube to move
- finally, the cube stops moving

Preference 2: 'throwAttempt'
This preference is satisfied when:
- first, the agent picks up a dodgeball
- next, the agent throws the dodgeball
- finally, the dodgeball stops moving

### TEMPLATED DESCRIPTION:
Preference 1: 'dodgeballsInPlace'
The variables required by this preference are:
- ?d: of type dodgeball
- ?h: of type hexagonal_bin

This preference is satisfied when:
- in the final game state, ?d is inside of ?h

Preference 2: 'blocksInPlace'
The variables required by this preference are:
- ?c: of type cube_block
- ?s: of type shelf

This preference is satisfied when:
- in the final game state, (?s is adjacent to west_wall) and (?c is on ?s)

Preference 3: 'laptopAndBookInPlace'
The variables required by this preference are:
- ?o: of type laptop or book
- ?s: of type shelf

This preference is satisfied when:
- in the final game state, (?o is on ?s)

Preference 4: 'smallItemsInPlace'
The variables required by this preference are:
- ?o: of type cellphone or key_chain
- ?d: of type drawer

This preference is satisfied when:
- in the final game state, (?o is inside of ?d)

Preference 5: 'itemsTurnedOff'
The variables required by this preference are:
- ?o: of type main_light_switch, desktop, or laptop

This preference is satisfied when:
- in the final game state, (it's not the case that ?o is toggled on)

### NATURAL LANGUAGE DESCRIPTION:
Preference 1: 'dodgeballsInPlace'
This preference is satisfied when:
- in the final game state, a dodgeball is inside of the hexagonal bin

Preference 2: 'blocksInPlace'
This preference is satisfied when:
- in the final game state, a cube block is on the shelf next to the west wall

Preference 3: 'laptopAndBookInPlace'
This preference is satisfied when:
- in the final game state, a laptop or a book are on a shelf

Preference 4: 'smallItemsInPlace'
This preference is satisfied when:
- in the final game state, a cellphone or a key chain are inside of a drawer

Preference 5: 'itemsTurnedOff'
This preference is satisfied when:
- in the final game state, the main light switch, the desktop, or the laptop is turned off

Now, convert the following description:
### TEMPLATED DESCRIPTION:
{0}"""

TERMINAL_STAGE_1_TO_STAGE_2_PROMPT = """Your task is to convert a templated description of a game's terminal conditions into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:

### TEMPLATED DESCRIPTION:
The game ends when the number of times 'throwAttempt' has been satisfied with different objects is greater than or equal to 2

### NATURAL LANGUAGE DESCRIPTION:
The game ends when 'throwAttempt' has been satisfied twice with different objects.

### TEMPLATED DESCRIPTION:
The game ends when ((total-time) is greater than or equal to 180) or ((total-score) is greater than or equal to 50)

### NATURAL LANGUAGE DESCRIPTION:
The game ends when 180 seconds have elapsed or the player has scored at least 50 points.

Now, convert the following description:
### TEMPLATED DESCRIPTION:
{0}"""

SCORING_STAGE_1_TO_STAGE_2_PROMPT= """Your task is to convert a templated description of a game's scoring conditions into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:

### TEMPLATED DESCRIPTION:
At the end of the game, the player's score is the sum of (the product of (10) and (the number of times 'thrownBallReachesEnd' has been satisfied)), (the product of (negative 5) and (the number of times 'thrownBallHitsBlock' has been satisfied with specific variable types red)), (the product of (negative 3) and (the number of times 'thrownBallHitsBlock' has been satisfied with specific variable types green)), (the product of (negative 3) and (the number of times 'thrownBallHitsBlock' has been satisfied with specific variable types pink)), (negative the number of times 'thrownBallHitsBlock' has been satisfied with specific variable types yellow), and (negative the number of times 'thrownBallHitsBlock' has been satisfied with specific variable types purple)

### NATURAL LANGUAGE DESCRIPTION:
At the end the game, the player gets 10 points for each time 'thrownBallReachesEnd' has been satisfied. They lose 5 points for each time 'thrownBallHitsBlock' has been satisfied with a red block, 3 points for each time 'thrownBallHitsBlock' has been satisfied with a green block, 3 points for each time 'thrownBallHitsBlock' has been satisfied with a pink block, and 1 point for each time 'thrownBallHitsBlock' has been satisfied with a yellow or purple block.

### TEMPLATED DESCRIPTION:
At the end of the game, the player's score is the sum of (the product of (5) and (the sum of (the number of times 'dodgeballsInPlace' has been satisfied with different objects), (the number of times 'blocksInPlace' has been satisfied with different objects), (the number of times 'laptopAndBookInPlace' has been satisfied with different objects), and (the number of times 'smallItemsInPlace' has been satisfied with different objects))) and (the product of (3) and (the number of times 'itemsTurnedOff' has been satisfied with different objects))

### NATURAL LANGUAGE DESCRIPTION:
At the end of the game, the player gets 5 points for every object used to satisfy 'dodgeballsInPlace', 5 points for every object used to satisfy 'blocksInPlace', 5 points for every object used to satisfy 'laptopAndBookInPlace', and 5 points for every object used to satisfy 'smallItemsInPlace'. They also get 3 points for every object used to satisfy 'itemsTurnedOff'.

Now, convert the following description:
### TEMPLATED DESCRIPTION:
{0}"""


### Additional prompt considerations
'''
At the end of the game, the player's score is the sum of (the product of (3), (the number of times 'throwAttempt' has been satisfied is equal to 1), and (whether 'throwOverRamp' has been satisfied at least once)); 
(the product of (2), (the number of times 'throwAttempt' has been satisfied is equal to 2), and (whether 'throwOverRamp' has been satisfied at least once)); 
and (the product of (the number of times 'throwAttempt' has been satisfied is greater than or equal to 3) and (whether 'throwOverRamp' has been satisfied at least once))

--> 3 points if you do it in one try, 2 points if you do it in two tries, 1 point if you do it in three or more tries
'''