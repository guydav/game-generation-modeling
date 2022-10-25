
; 6172feb1665491d1efbce164-0 (medium-objects-room-v1)
; SETUP: Place small ramp in front of bin where there is space in the room.
; GAMEPLAY: Take ball (any), and try to get it into the bin by rolling it hard (and straight) enough to pass the small ramp. When/if the bin hits over, game over. 
; SCORING: 1 point per successful hit. 
; DIFFICULTY: 1
(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)  ; 0
(:setup (and 
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
        (game-conserved (< (distance ?h ?r) 1))
    )
))
(:constraints (and 
    (preference throwToRampToBin
        (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?b)) 
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?r)
                ) 
                (once  (and (in ?h ?b) (not (in_motion ?b)))) 
            )
        )
    )
    (preference binKnockedOver
        (exists (?h - hexagonal_bin) 
            (then 
                (hold (and (not (touch agent ?h)) (not (agent_holds ?h))))
                (once (not (object_orientation ?h upright)))
            )
        )
    )
))
(:terminal (>= (count-once binKnockedOver) 1)
)
(:scoring maximize (count-nonoverlapping throwToRampToBin)
))

; 1 is invalid

; 5f77754ba932fb2c4ba181d8-2 (many-objects-room-v1)
; SETUP: Open the top drawer beside your bed.
; GAMEPLAY: First you pick up a dodgeball or a golf ball and stand right beside the room leaving out of the room. After that you throw either the dodgeball or the golf ball to the bin or the top drawer that we opened earlier. The game ends when you have thrown all the balls (six in total, 3 golfballs and 3 dodgeballs) and then you tally up your points.
; SCORING: To score in this game you just have to throw any of the balls either to top drawer or the bin. 
; If you get a golf ball in the bin you get only 1 point, but if you suceed to get a dodgeball in you get 2 points.
; The other method to score points is to throw in the golfballs to the top drawer. If you get the golf ball you get 3 points. 
; But if you miss any of the throws to either the bin or the drawer you get a reduction of 1 point.
; DIFFICULTY: 1
(define (game 5f77754ba932fb2c4ba181d8-2) (:domain many-objects-room-v1)  ; 2
(:setup (and 
    (game-conserved (open top_drawer))
))
(:constraints (and 
    (forall (?b - (either dodgeball golfball) ?t - (either top_drawer hexagonal_bin))
        (preference throwToDrawerOrBin
            (then 
                (once (and (agent_holds ?b) (adjacent agent door)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (not (in_motion ?b)) (in ?t ?b)))
            )
        )
    )
    (preference throwAttempt
        (exists (?b - (either dodgeball golfball))
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal (>= (count-once-per-objects throwAttempt) 6)
)
(:scoring maximize (+ 
    (count-once-per-objects throwToDrawerOrBin:golfball:hexagonal_bin)
    (* 2 (count-once-per-objects throwToDrawerOrBin:dodgeball:hexagonal_bin))
    (* 3 (count-once-per-objects throwToDrawerOrBin:golfball:top_drawer))
    (+ (count-once-per-objects throwToDrawerOrBin) (- (count-once-per-objects throwAttempt)))  ; as a way to encode -1 for each missed throw
)))

; 3 says "figures", but their demonstration only uses blocks, so I'm guessing that's what they meant

; 614b603d4da88384282967a7-3 (many-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: Create a tower with the largest number of figures available.
; SCORING: Each level of the tower will count as 1 point
; DIFFICULTY: 3
(define (game 614b603d4da88384282967a7-3) (:domain many-objects-room-v1)  ; 3
(:constraints (and 
    (forall (?b - building) 
        (preference blockInTowerAtEnd (exists (?l - block)
            (at-end (in ?b ?l))
        ))
    )
))
(:scoring maximize (+
    (count-maximal-once-per-objects blockInTowerAtEnd)
)))

; 4 is invalid -- woefully underconstrained

; 5bc79f652885710001a0e82a-5 (few-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: Throwing w Dogball to the can form 1 meter distance
; SCORING: 1 Dogball in the bin = 1 point
; DIFFICULTY: 1
(define (game 5bc79f652885710001a0e82a-5) (:domain few-objects-room-v1)  ; 5

(:constraints (and 
    (preference throwBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (= (distance agent ?h) 1)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring maximize (count-nonoverlapping throwBallToBin)
))

; 614dec67f6eb129c3a77defd-6 (medium-objects-room-v1)
; SETUP: Position the in front of the bed. Remove pillow and teddybear from the bed.
; GAMEPLAY: Try to trow the balls inside the bin from other side of the room. 
; SCORING: Dodge ball is 10 points. Basketball is 20 points. Beach ball is 30 points. Every miss is -1 point. 
; DIFFICULTY: 1
(define (game 614dec67f6eb129c3a77defd-6) (:domain medium-objects-room-v1)  ; 6
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (adjacent ?h bed)))
    (forall (?x - (either teddy_bear pillow)) (game-conserved (not (on bed ?x))))
))
(:constraints (and 
    (forall (?b - ball)
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then 
                    (once (and (agent_holds ?b) (adjacent agent desk)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            )
        )
    )
    (preference failedThrowToBin
        (exists (?b - ball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?b) (adjacent agent desk)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (not (in_motion ?b)) (not (in ?h ?b))))
            )
        )
    )
))
(:scoring maximize (+
    (* 10 (count-nonoverlapping throwBallToBin:dodgeball))
    (* 20 (count-nonoverlapping throwBallToBin:basketball))
    (* 30 (count-nonoverlapping throwBallToBin:beachball))
    (* (- 1) (count-nonoverlapping failedThrowToBin))
)))

; 7 is invalid -- vastly under-constrained -- I could probably make some guesses but leaving alone

; 615b40bb6cdb0f1f6f291f45-8 (few-objects-room-v1)
; SETUP: The room needs to have a completely open space in the middle. There should be nothing obstructing the way of the player.
; GAMEPLAY: You need the dodgeballs and the curved ramp. You would put the curved ramp on one end and then stand on the other end. you would then roll the ball hard enough to get over the curved ramp and would get a point when it goes over.
; SCORING: To score in my game, the ball needs to get over the ramp. If you manage to get the ball over on your first attempt that's 3 points, the second is 2 points and anything after that is 1 point.
; DIFFICULTY: 4
(define (game 615b40bb6cdb0f1f6f291f45-8) (:domain few-objects-room-v1)  ; 8
(:setup (and 
    (exists (?c - curved_wooden_ramp)
        (game-conserved (on floor ?c))
    )
))
(:constraints (and 
    (preference throwOverRamp  ; TODO: does this quanitfy over reasonably?
        (exists (?d - dodgeball ?c - curved_wooden_ramp)
            (then 
                (once (and 
                    (agent_holds ?d) 
                    (< (distance_side ?c front agent) (distance_side ?c back agent))
                ))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                ) 
                (once (and 
                    (not (in_motion ?d)) 
                    (< (distance_side ?c back ?d) (distance_side ?c front ?d))  
                ))
            )
        )
    )
    (preference throwAttempt
        (exists (?b - ball)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal (>= (count-once throwOverRamp) 1)
)
(:scoring maximize (+
    (* 3 (= (count-nonoverlapping throwAttempt) 1) (count-once throwOverRamp))
    (* 2 (= (count-nonoverlapping throwAttempt) 2) (count-once throwOverRamp))
    (* (>= (count-nonoverlapping throwAttempt) 3) (count-once throwOverRamp))
)))

; Taking the first game this participant provided

; 615452aaabb932ada88ef3ca-9 (many-objects-room-v1)
; SETUP: I got 3 Different ideas. The first 2 can be setup by yourself. 
; 1st one is a simple basketball game using the bin as a basket and the dodgeballs as bastketballs put the bin on one end of the room or on the bed (for a higher difficulty) and use the other end of the room as a marker from where you throw. 
; GAMEPLAY: 1. Basketball 
; SCORING: 1. 10 tries 1 Point if the ball lands in the Bin 3 Points if it lands in the Bin without touching the floor before. 
; DIFFICULTY: 1
(define (game 615452aaabb932ada88ef3ca-9) (:domain many-objects-room-v1)  ; 9
(:setup (and 
    (exists (?h - hexagonal_bin)
        (game-conserved (or
            (on bed ?h)
            (exists (?w - wall) (adjacent ?w ?h))
        ))
    )        
))
(:constraints (and 
    (preference throwBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and 
                    (agent_holds ?d)
                    (or 
                        (on bed ?h)
                        (exists (?w1 ?w2 - wall) (and (adjacent ?w1 ?h) (adjacent ?w2 agent) (opposite ?w1 ?w2)))
                    )    
                ))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    ; TODO: one could argue that these could be specified by providing another predicate in scoring
    ; so for example, say something like, if this predicate (bounce) also happens at some point during 
    ; the preference, you get additional or fewer points
    ; TODO: is that something we want to do? would this be useful anywhere else?
    (preference bounceBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and 
                    (agent_holds ?d)
                    (or 
                        (on bed ?h)
                        (exists (?w1 ?w2 - wall) (and (adjacent ?w1 ?h) (adjacent ?w2 agent) (opposite ?w1 ?w2)))
                    )    
                ))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch floor ?d)    
                ) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring maximize (+
    (count-nonoverlapping bounceBallToBin)
    (* 3 (count-nonoverlapping throwBallToBin))
)))

; 57aa430b4cda6e00018420e9-10 (medium-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: Throw teddy onto pillow 10 times
; SCORING: Score 1 point for everytime teddy lands on pillow
; DIFFICULTY: 2
(define (game 57aa430b4cda6e00018420e9-10) (:domain medium-objects-room-v1)  ; 10
(:constraints (and 
    (preference throwTeddyOntoPillow
        (exists (?t - teddy_bear ?p - pillow)
            (then 
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t))) 
                (once (and (not (in_motion ?t)) (on ?p ?t)))
            )
        )
    )
    (preference throwAttempt
        (exists (?t - teddy_bear)
            (then 
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t))) 
                (once (not (in_motion ?t)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 10)
)
(:scoring maximize (count-nonoverlapping throwTeddyOntoPillow)
))

; 5d29412ab711e9001ab74ece-11 (many-objects-room-v1)
; SETUP: No preparation required
; GAMEPLAY: Play a game of tower building using colored blocks. Your objective is to build highest possible tower that is stable, starting with any single block and ending with any pyramid block. No other block then first one is allowed to touch floor or walls or any non block objects in the room.
; SCORING: Your score is number of blocks in tower including first and last (pyramid block)
; DIFFICULTY: 4
(define (game 5d29412ab711e9001ab74ece-11) (:domain many-objects-room-v1)  ; 11
(:constraints (and 
    (forall (?b - building) (and 
        (preference baseBlockInTowerAtEnd (exists (?l - block)
            (at-end (and
                (in ?b ?l)  
                (on floor ?l)
            ))
        ))
        (preference blockOnBlockInTowerAtEnd (exists (?l - block)
            (at-end
                (and 
                    (in ?b ?l)
                    (not (exists (?o - game_object) (and (not (type ?o block)) (touch ?o ?l))))
                    (not (on floor ?l))
                )
            )
        )) 
        (preference pyramidBlockAtopTowerAtEnd (exists (?p - pyramid_block)
            (at-end
                (and
                    (in ?b ?p)   
                    (not (exists (?l - block) (on ?p ?l)))
                    (not (exists (?o - game_object) (and (not (type ?o block)) (touch ?o ?p))))
                )
            )
        )) 
    ))
))
(:scoring maximize (* 
    (count-maximal-once pyramidBlockAtopTowerAtEnd)
    (count-maximal-once baseBlockInTowerAtEnd)
    (+ 
        (count-maximal-once baseBlockInTowerAtEnd)
        (count-maximal-once-per-objects blockOnBlockInTowerAtEnd)   
    )     
)))

; 12 requires quantifying based on position -- something like

; 613bb29f16252362f4dc11a3-12 (medium-objects-room-v1)
; SETUP: You put the bin in the midle of the room and prepare the dodgeball and the wooden ramp.
; GAMEPLAY: You position the wooden ramp in different ways depending. Then You have to throw the ball at it and it should bounce into the bin. 
; SCORING: Each "wooden ramp" position You make You have 5 tries, if You get it right You get a point. Each game You choose 5 different Ramp positions. So the maximum of Points is 5. One for each Ramp possition
; DIFFICULTY: 4
(define (game 613bb29f16252362f4dc11a3-12) (:domain medium-objects-room-v1)  ; 12
(:setup (and 
    (exists (?h - hexagonal_bin)
        (game-conserved (< (distance ?h room_center) 1))
    )
))
(:constraints (and 
    (preference throwToRampToBin
        (exists (?r - triangular_ramp ?d - dodgeball ?h - hexagonal_bin) 
            (then 
                (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?r ?d)
                ) 
                (once  (and (in ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
            )
        )
    )
))
(:scoring maximize
    (count-unique-positions throwToRampToBin)
))

; 616e5ae706e970fe0aff99b6-13 (many-objects-room-v1)
; SETUP: you put the small ramp towards the center of the room about 1 meter away from the bin facing it
; GAMEPLAY: you roll the golf balls and the dodgeballs towards the small ramp aiming to put them in the bin.
; you sit down in front of the door and try to use the correct amount of power and aim precisely so that the balls go off the ramp into the air and then into the bin
; SCORING: dodgeballs count as 3 points and golfballs count as 6.
; DIFFICULTY: 3
(define (game 616e5ae706e970fe0aff99b6-13) (:domain many-objects-room-v1)  ; 13
(:setup (and 
    (exists (?h - hexagonal_bin ?r - triangular_ramp) (game-conserved 
        (and
            (< (distance ?h ?r) 1)
            (< (distance ?r room_center) 0.5)
        )
    ))
))
(:constraints (and 
    (forall (?d - (either dodgeball golfball))
        (preference throwToRampToBin
            (exists (?r - triangular_ramp ?h - hexagonal_bin) 
                (then 
                    (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
                    (hold-while 
                        (and (not (agent_holds ?d)) (in_motion ?d))
                        (touch ?r ?d)
                    ) 
                    (once (and (in ?h ?d) (not (in_motion ?d)))) ; touches ramp before in bin
                )
            )
        )
    )
))
(:scoring maximize (+
    (* 6 (count-nonoverlapping throwToRampToBin:dodgeball))
    (* 3 (count-nonoverlapping throwToRampToBin:golfball))
)))

; 609c15fd6888b88a23312c42-14 (medium-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: Player has to throw dodgeballballs, basketball and beachball into the bin. Player has to stand on the carpet.
; SCORING: Every scored ball is 1 point. Every 3 balls scored in a row are 1 extra poiunt for the player. Player has 10 throws.
; DIFFICULTY: 1
(define (game 609c15fd6888b88a23312c4-14) (:domain medium-objects-room-v1)  ; 14
(:constraints (and 
    (preference throwInBin
        (exists (?b - ball ?h - hexagonal_bin)
            (then 
                (once (and (on rug agent) (agent_holds ?b)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        )
    )
    (preference throwAttempt
        (exists (?b - ball)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 10)
)
(:scoring maximize (count-nonoverlapping throwInBin)
    ; TODO: how do we want to quantify streaks? some number of one preference without another preference?
))

; 5f5d6c3cbacc025bf0a03440-15 (few-objects-room-v1)
; SETUP: First the bin should be turned upside down and placed on the side of the bed; Subsequently, the cubeblocks must be placed in a pyramidal way on top of the bin.
; GAMEPLAY: The game that can be played is to use the dodgeball to throw the objectives placed on top of the bin (cubeblocks); all this must be done sitting from the chair at the other end of the room; being two dodgeballs there are only two chances to achieve the most points.
; SCORING: The scoring system consists in that each cubeblock that falls through the dodgeball will be equal to 1 point (6 cubeblocks equal 6 points); With only two dodgeballs, you only have 2 chances to get points; In case of throwing the 6 cubeblocks on the first shot, they must be replaced to get more points with the last dodgeball.
; DIFFICULTY: 2
(define (game 5f5d6c3cbacc025bf0a03440-15) (:domain few-objects-room-v1)  ; 15
(:setup (and
    (exists (?h - hexagonal_bin ?b - building) (and 
        (game-conserved (adjacent ?h bed))
        (game-conserved (object_orientation ?h upside_down))
        (game-optional (on ?h ?b)) ; optional since building might cease to exist in game
        (forall (?c - cube_block) (game-optional (in ?b ?c)))
        (exists (?c1 ?c2 ?c3 ?c4 ?c5 ?c6 - cube_block) (game-optional (and ; specifying the pyramidal structure
           (on ?h ?c1)
           (on ?h ?c2)
           (on ?h ?c3)
           (on ?c1 ?c4)
           (on ?c2 ?c5)
           (on ?c4 ?c6) 
        )))
    ))
))
(:constraints (and 
    (preference blockInTowerKnockedByDodgeball (exists (?b - building ?c - cube_block 
        ?d - dodgeball ?h - hexagonal_bin ?c - chair)
        (then
            (once (and 
                (agent_holds ?d)
                (adjacent agent ?c)
                (on ?h ?b)
                (in ?b ?c) 
            ))
            (hold-while (and (not (agent_holds ?d)) (in_motion ?d))
                (or 
                    (touch ?c ?d)
                    (exists (?c2 - cube_block) (touch ?c2 ?c))
                )
                (in_motion ?c)
            )
            (once (not (in_motion ?c)))
        )
    ))
    (preference throwAttempt
        (exists (?d - dodgeball)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:terminal
    (>= (count-once-per-objects throwAttempt) 2)
)
(:scoring maximize (count-once-per-objects blockInTowerKnockedByDodgeball)
))

; 616e4f7a16145200573161a6-16 (few-objects-room-v1)
; SETUP: closed blinds
; the curved ramp with the square boxes on the side
; the bin after the ramp
; GAMEPLAY: roll the dodge ball onto the bin across the ramp with the cube blocks around to direct the ball onto the ramp to the bin
; SCORING: if a ball enters the box its a point
; DIFFICULTY: 3
(define (game 616e4f7a16145200573161a6-16) (:domain few-objects-room-v1)  ; 16
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin ?b1 ?b2 ?b3 ?b4 - block) 
        (game-conserved (and
            (adjacent_side ?h front ?c back)
            (on floor ?b1)
            (adjacent_side ?h left ?b1)
            (on ?b1 ?b2)
            (on floor ?b3)
            (adjacent_side ?h right ?b3)
            (on ?b3 ?b4)
        ))
    )
))
(:constraints (and 
    (preference rollBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?d)) 
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) 
                    (on ?r ?d) 
                )
                (once (and (in ?h ?d) (not (in_motion ?d)))) 
            )
        ) 
    )
))
(:scoring maximize (count-nonoverlapping rollBallToBin)
))

; 18 is a dup of 17

; 613e4bf960ca68f8de00e5e7-17 (medium-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: The pieces on the shelf between the two windows must be stacked in such a way as to create a kind of castle. You have to create two equal castles, the colors are not important but if you choose certain colors points are added.
; To create each castle you have to follow this order of pieces (from bottom to top):
; - Bridge
; - Flat piece
; - Cylinder
; - Square
; - Pyramid
; SCORING: For each castle built in the correct order of pieces will be awarded 10 points.
; If the castles follow this order of colors extra points will be awarded:
; First castle:
; - Green bridge
; - Glat yellow piece
; - Yellow cylinder
; - Green square
; - Orange pyramid
; Second castle: 
; - Bridge color wood
; - Flat gray piece
; - Cylinder color wood
; - Blue cylinder
; - Red pyramid
; If 3 of the pieces have the correct color order (in the corresponding village) 5 points per castle will be awarded, if 4 colors are good it will be 7 points per castle and if all colors are good, 10 points per castle. The maximum score is 40 points.
; DIFFICULTY: 0
(define (game 613e4bf960ca68f8de00e5e7-17) (:domain medium-objects-room-v1)  ; 17/18
(:constraints (and 
    (preference castleBuilt (exists (?b - bridge_block ?f - flat_block ?t - tall_cylindrical_block ?c - cube_block ?p - pyramid_block)
        (at-end
            (and 
                (on ?b ?f)
                (on ?f ?t)
                (on ?t ?c)
                (on ?c ?p)
            )
        )
    ))
))
(:scoring maximize (+ 
    (* 10 (count-once-per-objects castleBuilt))
    ; (* 4 (or 
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?c - green_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (or 
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))) 
    ; (* 4 (or 
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?c - blue_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - gray_flat_block ?t - brown_tall_cylindrical_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (or 
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))) 
)))

; 60e93f64ec69ecdac3107555-19 (medium-objects-room-v1)
; SETUP: The player should move basketball, beachball and dodgeball closer to the door. It is not obligatory but would make playing the game more comfortable.
; GAMEPLAY: The player stands next to the door facing the room. He has three balls - basketball, beachball and dodgeball - next to him, as well as dogbed and bin in the opposite corners of the room. The game is about trying to throw each ball in either the dogbed or bin to try to get the most points.
; SCORING: If the ball was succesfully thrown into the bin, this would give 3 points. If the ball was thrown into the dogbed, this gives the player 2 points, but if the ball stops on the corner of the dogbed - that gives 1 point. If the ball which the player succesfully threw was basketball, it gives additionally 1 point, if beachball - 2 points, if dodgeball - 3 points. The score of each throw consists of points received from succesfully throwing into specific item plus the points that gives the specific ball. Final game score consists of points from each throw and range from 0 to 15 points. 
; DIFFICULTY: 2
(define (game 60e93f64ec69ecdac3107555-19) (:domain medium-objects-room-v1)  ; 19
(:setup (and
    (forall (?b - ball)
        (game-optional (< (distance ?b door) 1))
    )
))
(:constraints (and 
    (forall (?b - ball ?t - (either doggie_bed hexagonal_bin))
        (preference ballThrownIntoTarget
            (then 
                (once (and (agent_holds ?b) (< (distance agent door) 1)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (in ?t ?b) (not (in_motion ?b))))
            )
        )
    )
    (forall (?b - ball)
        (preference ballThrownOntoTarget
            (exists (?t - doggie_bed) 
                (then 
                    (once (and (agent_holds ?b) (< (distance agent door) 1)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                    (once (and (on ?t ?b) (not (in_motion ?b))))
                )
            )
        )
    )
    (preference throwAttempt
        (exists (?b - ball)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal
    (>= (count-once-per-objects throwAttempt) 3)
)
(:scoring maximize (+ 
    (* 6 (count-once-per-objects ballThrownIntoTarget:dodgeball:hexagonal_bin))
    (* 5 (count-once-per-objects ballThrownIntoTarget:beachball:hexagonal_bin))
    (* 4 (count-once-per-objects ballThrownIntoTarget:basketball:hexagonal_bin))
    (* 5 (count-once-per-objects ballThrownIntoTarget:dodgeball:doggie_bed))
    (* 4 (count-once-per-objects ballThrownIntoTarget:beachball:doggie_bed))
    (* 3 (count-once-per-objects ballThrownIntoTarget:basketball:doggie_bed))
    (* 5 (count-once-per-objects ballThrownOntoTarget:dodgeball))
    (* 4 (count-once-per-objects ballThrownOntoTarget:beachball))
    (* 3 (count-once-per-objects ballThrownOntoTarget:basketball))
)))

; 5e2df2855e01ef3e5d01ab58-20 (medium-objects-room-v1)
; SETUP: No setup needed.
; GAMEPLAY: To play my game, you will use the building blocks on the shelf, to build as high of a tower as possible. You can play it safe and use less blocks, but for less points, or a higher tower for more points. When you have built as high as you dare, you will place yourself on the far side of the room with the red dodgeball. Then, you will throw the ball at the tower and try to knock down as many blocks as you can.
; SCORING: To score my game, you will get 1 point for each block you use to build the tower. If your tower falls over while building you will get -1 points. Additionally, you will get 1 point for each block you manage to knock over with the dodgeball in phase 2.
; DIFFICULTY: 2
(define (game 5e2df2855e01ef3e5d01ab58-20) (:domain medium-objects-room-v1) ; 20
(:constraints (and 
    (forall (?b - building) (and  
        (preference blockInTowerAtEnd (exists (?l - block)
            (at-end
                (and 
                    (in ?b ?l)
                )
            )
        ))
        (preference blockInTowerKnockedByDodgeball (exists (?l - block ?d - dodgeball)
            (then
                (once (and (in ?b ?l) (agent_holds ?d)))
                (hold (and (in ?b ?l) (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (in ?b ?l) (touch ?d ?b)))
                (hold (in_motion ?l))
                (once (not (in_motion ?l)))
            )
        ))
    ))
    (preference towerFallsWhileBuilding (exists (?b - building ?l1 ?l2 - block)
        (then
            (once (and (in ?b ?l1) (agent_holds ?l2)))
            (hold-while 
                (and
                    (not (agent_holds ?l1)) 
                    (in ?b ?l1)
                    (or 
                        (agent_holds ?l2) 
                        (and (not (agent_holds ?l2)) (in_motion ?l2))
                    )
                )
                (touch ?l1 ?l2)
            )
            (hold (and 
                (in_motion ?l1)
                (not (agent_holds ?l1))
            ))
            (once (not (in_motion ?l1)))
        )
    ))
))
(:scoring maximize (+ 
    (count-maximal-once-per-objects blockInTowerAtEnd)
    (* 2 (count-maximal-once-per-objects blockInTowerKnockedByDodgeball))
    (* (- 1) (count-nonoverlapping towerFallsWhileBuilding))
)))

; 5c79bc94d454af00160e2eee-21 (few-objects-room-v1)
; SETUP: I moved a chair to the middle of the room, and turned it so it was facing sideways. I moved both balls; one to the bin and one to the bed, to show where two of the goals are.
; GAMEPLAY: To play my game, you start at the table at the far-side of the room. The goal is to throw the ball in the bin, but you also get points (although less points) if the ball lands on the chair or the bed, since those are easier targets. It's like a bedroom version of basketball.
; SCORING: If you score the ball into the bin - 5 Points
; If you score the ball onto the bed - 1 Point
; If you score the ball onto the chair - 1 Point
; If your ball ends up on the floor or on any other object - Lose 2 points.
; You win when you gain 10 points. You lose if your points get down to negative 5 points.
; DIFFICULTY: 1
(define (game 5c79bc94d454af00160e2eee-21) (:domain few-objects-room-v1)  ; 21
(:setup (and 
    (exists (?c - chair) (game-conserved (and 
        (< (distance ?c room_center) 1)
        (not (faces ?c desk))
        (not (faces ?c bed))
    ))) 
))
(:constraints (and 
    (preference ballThrownToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    (preference ballThrownToBed
        (exists (?d - dodgeball)
            (then 
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (on bed ?d)))
            )
        )
    )
    (preference ballThrownToChair
        (exists (?d - dodgeball ?c - chair)
            (then 
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (on ?c ?d) (< (distance ?c room_center) 1)))
            )
        )
    )
    (preference ballThrownMissesEverything
        (exists (?d - dodgeball)
            (then 
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and 
                    (not (in_motion ?d)) 
                    (not (exists (?h - hexagonal_bin) (in ?h ?d)))
                    (not (on bed ?d))
                    (not (exists (?c - chair) (and (on ?c ?d) (< (distance ?c room_center) 1))))
                ))
            )
        )
    )
))
(:terminal
    (>= (total-score) 10)
)
(:scoring maximize (+ 
    (* 5 (count-nonoverlapping ballThrownToBin))
    (count-nonoverlapping ballThrownToBed)
    (count-nonoverlapping ballThrownToChair)
    (* (- 1) (count-nonoverlapping ballThrownMissesEverything))
)))

; 60d432ce6e413e7509dd4b78-22 (medium-objects-room-v1)
; SETUP: To prepare the room for the game, the bin must be placed at the foot of the desk.
; A Beachball, a BasketBall and a Dodgeball must be near the player. The player must stand on the carpet.
; The Desktop, Laptop and any other objects placed on the desk should be moved to avoid hazards.
; GAMEPLAY: The primary goal of my game is to throw balls into a bin across the room. Depending on the distance and the type of ball thrown, you may gain more points. You stand on the carpet and aim to toss balls into the bin on the opposite side of the room near the desk. The player has 8 attempts to reach the highest score possible.
; SCORING: If the player uses a Dodgeball and throws it into the bin, they receive 1 point. If the player uses a BasketBall and throws it into the bin, they receive 2 points. If the player uses a Beachball and it lands on top of the bin, they receive 3 points. Bonus points can be earned from the game depending on where the ball is thrown. If the ball is thrown on the red line of the carpet and the ball enters the bin, the player recieves 1 extra point. If the player throws the ball from the pink square on the carpet and the ball enters the bin, the player recieves 2 extra points. If the same is done on the yellow square on the carpet, the player receives 3 points. This incentives players to shoot from farther distances to gain points. Easier and smaller balls award less points, and bigger balls award more points. Since there are 8 attempts or 8 turns, the highest possible amount of points earned could be 48 using the Beachball on the yellow carpet tile. If the player misses their shot, they earn 0 points and move onto the next attempt.
; DIFFICULTY: 3
(define (game 60d432ce6e413e7509dd4b78-22) (:domain medium-objects-room-v1)  ; 22
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (adjacent bed ?h)))
    (forall (?b - ball) (game-optional (on rug ?b)))
    (game-optional (not (exists (?g - game_object) (on desk ?g))))
))
(:constraints (and 
    (forall (?b - ball ?c - (either red yellow pink))
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then 
                    (once (and (agent_holds ?b) (on rug agent) (rug_color_under agent ?c)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            )
        )
    )
    (preference throwAttempt
        (exists (?b - ball)
            (then 
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 8)
)
(:scoring maximize (+ 
    (* 2 (count-nonoverlapping throwBallToBin:dodgeball:red))
    (* 3 (count-nonoverlapping throwBallToBin:basketball:red))
    (* 4 (count-nonoverlapping throwBallToBin:beachball:red))
    (* 3 (count-nonoverlapping throwBallToBin:dodgeball:pink))
    (* 4 (count-nonoverlapping throwBallToBin:basketball:pink))
    (* 5 (count-nonoverlapping throwBallToBin:beachball:pink))
    (* 4 (count-nonoverlapping throwBallToBin:dodgeball:yellow))
    (* 5 (count-nonoverlapping throwBallToBin:basketball:yellow))
    (* 6 (count-nonoverlapping throwBallToBin:beachball:yellow))
)))

; 61267978e96853d3b974ca53-23 (few-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: Throw a dodgeball into the bin to score points.
; SCORING: Everytime you score a dodgeball into the bin you get 1 point and you lose 1 point every 5 throws.
; DIFFICULTY: 1
(define (game 61267978e96853d3b974ca53-23) (:domain few-objects-room-v1)  ; 23

(:constraints (and 
    (preference throwBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    (preference throwAttempt
        (exists (?d - dodgeball)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:scoring maximize (+
    (count-nonoverlapping throwBallToBin)
    (- (/ (count-nonoverlapping throwAttempt) 5))
)))

; 5996d2256b939900012d9f22-24 (few-objects-room-v1)
; SETUP: The bin has to go on top of a chair to make it more challenging.
; GAMEPLAY: The idea is to get each of the coloured balls into the bin.
; SCORING: First, the player must choose a place to stand on the multi-coloured rug.
; Each of the balls is worth a different amount. So if the player gets the the blue ball in, they score 5 points, and if they get the pink ball in they score 10 points.
; If the player stands on the red part of the rug, they can 1x the value of the ball they scored a goal with.
; If the player scores from the pink part, they score 2x.
; They score 3x from the orange or green parts.
; 4x the amount from the purple or yellow parts.
; If a ball is missed, they score nothing.
; The game ends when they get to 300 points.
; DIFFICULTY: 4
(define (game 5996d2256b939900012d9f22-24) (:domain few-objects-room-v1)  ; 24
(:setup (and 
    (exists (?c - chair ?h - hexagonal_bin) (game-conserved (on ?c ?h)))
))
(:constraints (and 
    (forall (?d - dodgeball ?c - color)
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then 
                    (once (and (agent_holds ?d) (on rug agent) (rug_color_under agent ?c)))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                    (once (and (not (in_motion ?d)) (in ?h ?d)))
                )
            )
        )
    )
))
(:terminal
    (>= (total-score) 300)
)
(:scoring maximize (+ 
    (* 5 (count-nonoverlapping throwBallToBin:blue_dodgeball:red))
    (* 10 (count-nonoverlapping throwBallToBin:pink_dodgeball:red))
    (* 10 (count-nonoverlapping throwBallToBin:blue_dodgeball:pink))
    (* 20 (count-nonoverlapping throwBallToBin:pink_dodgeball:pink))
    (* 15 (count-nonoverlapping throwBallToBin:blue_dodgeball:orange))
    (* 30 (count-nonoverlapping throwBallToBin:pink_dodgeball:orange))
    (* 15 (count-nonoverlapping throwBallToBin:blue_dodgeball:green))
    (* 30 (count-nonoverlapping throwBallToBin:pink_dodgeball:green))
    (* 20 (count-nonoverlapping throwBallToBin:blue_dodgeball:purple))
    (* 40 (count-nonoverlapping throwBallToBin:pink_dodgeball:purple))
    (* 20 (count-nonoverlapping throwBallToBin:blue_dodgeball:yellow))
    (* 40 (count-nonoverlapping throwBallToBin:pink_dodgeball:yellow))
)))

; 25 and 26 are the same participant and are invalid -- hiding games

; 606e4eb2a56685e5593304cd-27 (few-objects-room-v1)
; SETUP: Need to scatter the items around the room. Take the balls down from the shelf. Scatter the cubes around the room. Turn the lights and desktop computer on. 
; GAMEPLAY: The game would be called "Clean Up." You would need to find a space to put everything away. For instance, the dodgeballs would go in the bin. The small items, phone, keys, etc..., would all be put in the drawer of the bedside table. The book and laptop would be stored on the side shelves. The square bins would be stored on the shelves in the back of the room.
; SCORING: Every item in the room would have a value. For each item that is put away in the correct location, the player would receive 5 points. For every item that is turned off (to conserve electricity), the player would receive 3 points.
; DIFFICULTY: 3
(define (game 606e4eb2a56685e5593304cd-27) (:domain few-objects-room-v1)  ; 27
(:setup (and 
    (forall (?d - (either dodgeball cube_block)) (game-optional (not (exists (?s - shelf) (on ?s ?d)))))
    (game-optional (toggled_on main_light_switch))
    (game-optional (toggled_on desktop))
))
(:constraints (and 
    (preference dodgeballsInPlace 
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (at-end (in ?h ?d))
        )
    )
    (preference blocksInPlace
        (exists (?c - cube_block ?s - shelf)
            (at-end (and 
                (adjacent ?s west_wall)
                (on ?s ?c)
            ))
        )
    )
    (preference laptopAndBookInPlace
        (exists (?o - (either laptop book) ?s - shelf)
            (at-end (and 
                (adjacent ?s south_wall)
                (on ?s ?o)
            ))
        )
    )
    (preference smallItemsInPlace
        (exists (?o - (either cellphone key_chain) ?d - drawer)
            (at-end (and 
                (in ?d ?o)
            ))
        )
    )
    (preference itemsTurnedOff
        (exists (?o - (either main_light_switch desktop laptop))
            (at-end (and 
                (not (toggled_on ?o))
            ))
        )
    )
))
(:scoring maximize (+
    (* 5 (+
        (count-once-per-objects dodgeballsInPlace)
        (count-once-per-objects blocksInPlace)
        (count-once-per-objects laptopAndBookInPlace)
        (count-once-per-objects smallItemsInPlace)
    ))
    (* 3 (count-once-per-objects itemsTurnedOff))
)))

; 610aaf651f5e36d3a76b199f-28 (few-objects-room-v1)
; SETUP: the balls, and colorful rug and the cubes
; GAMEPLAY: lay the blocks on the colorful pat in a random order. Then I would grab the balls and turn around and toss the balls like in a bowling matter, avoiding the blocks. 
; SCORING: hit a block at level one the red part of pat (-5), hit on green and pink (-3) and (-1) at yellow and purple. if you make it to the end thats 10 points. you want to reach 50 points in 3 minuets. 
; DIFFICULTY: 2
(define (game 610aaf651f5e36d3a76b199f-28) (:domain few-objects-room-v1)  ; 28
(:setup (and 
    (forall (?c - cube_block) (game-conserved (on rug ?c)))
))
(:constraints (and 
    (forall (?c - color)
        (preference thrownBallHitsBlock
            (exists (?d - dodgeball ?b - cube_block)
                (then 
                    (once (and (agent_holds ?d) (not (on rug agent))))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                    (once (and (on rug ?b) (touch ?b ?d) (rug_color_under ?b ?c)))
                )
            )
        )
    )
    (preference thrownBallReachesEnd
            (exists (?d - dodgeball)
                (then 
                    (once (and (agent_holds ?d) (not (on rug agent))))
                    (hold-while 
                        (and 
                            (not (agent_holds ?d)) 
                            (in_motion ?d)
                            (not (exists (?b - cube_block) (touch ?d ?b)))    
                        )
                        (above rug ?d)
                    ) 
                    (once (or (touch ?d bed) (touch ?d west_wall)))
                )
            )
        )
))
(:terminal (or 
    (>= (total-time) 180)
    (>= (total-score) 50)
))
(:scoring maximize (+
    (* 10 (count-nonoverlapping thrownBallReachesEnd))
    (* (- 5) (count-nonoverlapping thrownBallHitsBlock:red))
    (* (- 3) (count-nonoverlapping thrownBallHitsBlock:green))
    (* (- 3) (count-nonoverlapping thrownBallHitsBlock:pink))
    (* (- 1) (count-nonoverlapping thrownBallHitsBlock:yellow))
    (* (- 1) (count-nonoverlapping thrownBallHitsBlock:purple))
)))

; 5bb511c6689fc5000149c703-29 (few-objects-room-v1)
; SETUP: The game starts as it is.
; GAMEPLAY: You need to place all movable objects on the bed. The pillow and blanket (which are already on) don't count. 
; The best score is 22.
; SCORING: Each object placed gives you 1 point.
; DIFFICULTY: 2
(define (game 5bb511c6689fc5000149c703-29) (:domain few-objects-room-v1)  ; 29
(:constraints (and 
    (preference objectOnBed
        (exists (?g - game_object)
            (at-end (and 
                (not (type ?g pillow))  
                (on bed ?g)
            ))
        )
    )
))
(:scoring maximize
    (count-nonoverlapping objectOnBed)
))


; 30 is invalid --  rather underdetermined, I could try, but it would take some guesswork

; 5b8c8e7d0c740e00019d55c3-31 (few-objects-room-v1)
; SETUP: The bin needs to be as far as it can from the bed. Close to the desk. The chair, that is in the middle can be moved for this to work. The cube blocks will be around the bin only to be realistic, this means the bin will not move if you hit it, the blocks will help the bin not to move.
; GAMEPLAY: The idea is to throw as many small objects inside the bin as you can. The clock, the cell phone, the mug, the keys, the cd, and probably the book and balls, if possible. You can only throw the objects standing on the coloured rug, you can't come any closer towards the bin. The red line on the rug is the limit.
; The objects you can throw will be prepared on the nightstand and on the bed. Once there are no objects on the bed and the nightstand, the game will be over.
; SCORING: Each object from the nightstand thrown inside the bin gives 1 point. Each object from the bed thrown inside the bin gives 2 points. 
; The more points the better.
; DIFFICULTY: 2
(define (game 5b8c8e7d0c740e00019d55c3-31) (:domain few-objects-room-v1)  ; 31
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (and 
        (adjacent desk ?h)
        (forall (?b - cube_block) (adjacent ?h ?b))
    )))
    (forall (?o - (either alarm_clock cellphone mug key_chain cd book ball))
        (game-optional (or 
            (on side_table ?o)
            (on bed ?o)   
        ))
    )
))
(:constraints (and 
    (forall (?s - (either bed side_table))
        (preference objectThrownFromRug
            (exists (?o - (either alarm_clock cellphone mug key_chain cd book ball) ?h - hexagonal_bin)
                (then
                    (once (on ?s ?o))
                    (hold (and (agent_holds ?o) (on rug agent)))
                    (hold (and (not (agent_holds ?o)) (in_motion ?o))) 
                    (once (and (not (in_motion ?o)) (in ?h ?o)))
                )
            )
        )
    )
))
(:scoring maximize (+
    (count-nonoverlapping objectThrownFromRug:side_table)
    (* 2 (count-nonoverlapping objectThrownFromRug:bed))
)))

; 56cb8858edf8da000b6df354-32 (many-objects-room-v1)
; SETUP: Take the bin and place it in any corner of the room. I used the corner that is on the bed. Also take the cube, cylinder and triangle blocks and make a pyramid (3-2-1) on the desk.
; GAMEPLAY: It's a mix of basketball and throwing balls in the pyramid of blocks to knock them over. I start of by trying to get as many balls as I can in the bin, to score enough points. Golf balls and dodgeballs score differently. Golf balls give 2 points, and dodgeballs give 1 point. You need to score 2 or more points in total, which can be done by a different combination of getting balls in the bin. You only have two tries for each of the 6 balls. After scoring the 2 points, you move to the other side of the room and get the remaining balls with you. You throw them and try to knock the pyramid over. For each block you knock over, you get 1 point. If you knock them all over and still have some balls remaining, you get 2 points for the dodgeballs and 1 for the golf balls that remain, and the total of the points from the balls and the points from the blocks knocked over is the final score.
; SCORING: I also explained it above but basically, at the end of the game, every block you knock over counts as 1 point, every remaining golf ball counts as 1 point, and every remaining dodge ball counts as 2 points.
; DIFFICULTY: 3
(define (game 56cb8858edf8da000b6df354-32) (:domain many-objects-room-v1)  ; 32
(:setup (and 
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - (either cube_block cylindrical_block pyramid_block)) (game-optional (and ; specifying the pyramidal structure
        (on desk ?b1)
        (on desk ?b2)
        (on desk ?b3)
        (on ?b1 ?b4)
        (on ?b2 ?b5)
        (on ?b4 ?b6) 
    )))
    (exists (?w1 ?w2 - wall ?h - hexagonal_bin) 
        (game-conserved (and
            (adjacent ?h ?w1)
            (adjacent ?h ?w2)   
        ))
    )
))
(:constraints (and 
    (forall (?b - (either dodgeball golfball)) 
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
    (preference blockInTowerKnocked (exists (?b - building ?c - (either cube_block cylindrical_block pyramid_block)
        ?d - (either dodgeball golfball))
        (then
            (once (and 
                (agent_holds ?d)
                (on desk ?b)
                (in ?b ?c) 
            ))
            (hold-while 
                (and (not (agent_holds ?d)) (in_motion ?d))
                (or 
                    (touch ?c ?d)
                    (exists (?c2 - (either cube_block cylindrical_block pyramid_block)) (touch ?c2 ?c))
                )
                (in_motion ?c)
            )
            (once (not (in_motion ?c)))
        )
    ))
    (forall (?d - (either dodgeball golfball))
        (preference throwAttempt
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )
    (forall (?d - (either dodgeball golfball))
        (preference ballNeverThrown
            (then
                (once (game_start))
                (hold (not (agent_holds ?d)))
                (hold (game_over))
            )
        )
    )
))
(:terminal (or 
    (> (count-maximal-nonoverlapping throwAttempt) 2)
    (>= (count-nonoverlapping throwAttempt) 12)
))
(:scoring maximize (* 
    (>=     
        (+
            (count-nonoverlapping ballThrownToBin:dodgeball)
            (* 2 (count-nonoverlapping ballThrownToBin:golfball))
        ) 
        2
    )
    (+
        (count-once-per-objects blockInTowerKnocked)
        (count-once-per-objects ballNeverThrown:golfball)
        (* 2 (count-once-per-objects ballNeverThrown:dodgeball))
    )
)))

; 614e1599db14d8f3a5c1486a-33 (many-objects-room-v1)
; SETUP: Bedside top drawer must be empty
; GAMEPLAY: How many items in the room can be stored simultaneously in the drawer
; SCORING: 1 point for each item that can be simultaneously stored in the drawer
; Item cannot be broken
; Items must fit when the drawer closes
; DIFFICULTY: 2
(define (game 614e1599db14d8f3a5c1486a-33) (:domain many-objects-room-v1)  ; 33
(:setup (and 
    (forall (?g - game_object) (game-optional 
        (not (in top_drawer ?g))   
    ))
))
(:constraints (and 
    (preference itemInClosedDrawerAtEnd (exists (?g - game_object)
        (at-end (and
            (in top_drawer ?g)
            (not (open top_drawer))
        ))
    ))
))
(:scoring maximize
    (count-once-per-objects itemInClosedDrawerAtEnd)
))

; 34 is invalid, another hiding game

; 615dd68523c38ecff40b29b4-35 (few-objects-room-v1)
; SETUP: Move the two balls on the shelf to the floor along with the ramp underneath the shelf. Move the bin to anywhere you'd like in the room.
; GAMEPLAY: You could shoot the balls into the bin in the room and play basketball. You could put the bin anywhere you would like into the room ie. the shelves, the desk, the bed, anywhere you would like. Shoot the balls into the bin.
; SCORING: Shoot as many balls as you like, every time you score you get 2 points but every time you miss you go down 1. If you hit the ball off the wall or another object, you get double the points which would be called a "trick shot." If you can make the book into the bin, you get 10 points. 1 time. You can get to 10 points, but if you go under 30, you lose. 10 wins.
; DIFFICULTY: 3
(define (game 615dd68523c38ecff40b29b4-35) (:domain few-objects-room-v1)  ; 35
(:constraints (and 
    (forall (?b - (either book dodgeball))
        (preference throwObjectToBin
            (exists (?h - hexagonal_bin)
                (then 
                    (once (agent_holds ?b))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b) (not (exists (?g - (either game_object floor wall)) (touch ?g ?b ))))) 
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            )
        )
    )
    (preference throwBallToBinOffObject
        (exists (?d - dodgeball ?h - hexagonal_bin ?g - (either game_object floor wall))
            (then 
                (once (agent_holds ?d))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?g ?d)
                ) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    (preference throwMissesBin
        (exists (?b - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (not (in_motion ?b)) (not (in ?h ?b))))
            )
        )
    )
))
(:terminal (or 
    (>= (total-score) 10)
    (<= (total-score) (- 30))
))
(:scoring maximize (+
    (count-nonoverlapping throwObjectToBin:dodgeball)
    (* 10 (count-once throwObjectToBin:book))
    (* 2 (count-nonoverlapping throwBallToBinOffObject))
    (* (- 1) (count-nonoverlapping throwMissesBin))
)))

; 5ef4c07dc8437809ba661613-36 (few-objects-room-v1)
; SETUP: You need to set up the bin on the bed and the two dodgeballs in the table. (you may need to remove the computers)
; GAMEPLAY: you have to throw the dogeballs from across the room and you need to try to get them in the bin. You have five tries so that you don't spend all day throwing dodgeballs.
; SCORING: whenever you get a point you put one of the blocks on the shelf. (on any of the two, it doesn't matter)
; DIFFICULTY: 3
(define (game 5ef4c07dc8437809ba661613-36) (:domain few-objects-room-v1)  ; 36
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (on bed ?h)))
    (forall (?d - dodgeball) (game-optional (on desk ?d)))
))
(:constraints (and 
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
                ; TODO: do we do anything about "whenever you get a point you put one of the blocks on the shelf. (on any of the two, it doesn't matter)"??
            )
        )
    )
    (preference throwAttempt
        (exists (?d - dodgeball)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 5)
)
(:scoring maximize
    (count-nonoverlapping throwToBin)
))

; 5fa45dc96da3af0b7dcba9a8-37 (many-objects-room-v1)
; SETUP: Pick up a dodgeball, put your back against the wall opposite the bin
; GAMEPLAY: Throw the ball using the appropriate amount of force to make it go inside the bin
; SCORING: 1 point per successful throw, 10 throws in a row, maximum of 10 points
; DIFFICULTY: 3
(define (game 5fa45dc96da3af0b7dcba9a8-37) (:domain many-objects-room-v1)  ; 37
(:constraints (and 
    (preference throwToBinFromOppositeWall
        (exists (?d - dodgeball ?h - hexagonal_bin ?w1 ?w2 - wall)
            (then 
                (once (and 
                    (agent_holds ?d) 
                    (adjacent agent ?w1)
                    (opposite ?w1 ?w2)
                    (adjacent ?h ?w2)
                ))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    (preference throwAttempt
        (exists (?d - dodgeball)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 10)
)
(:scoring maximize
    (count-nonoverlapping throwToBin)
))

; projected 38 onto the space of feasible games, but could also ignore

; 616abb33ebe1d6112545f76d-38 (medium-objects-room-v1)
; SETUP: To prepare the room I would leave what already is inside of it. Chairs, balls, bin, etc. But for the game what i would need would be balls, a bin, and maybe some paper from a printer
; GAMEPLAY: The game would consist in throwing the dodgeball to the bin, and try to put it inside. When i say the dodgeball, could also be the paper from printer in form of a ball, could be a small book, toys. The issue here is, if the player is stuck in the room, and would get bored, mind will always work its ways to stay out of boredom so it will create these simple types of games.
; SCORING: The scoring system would consist in if the dodgeball is big, if the player is able to put it inside the bin by throwing would give us something like 5 points. If there were smaller balls like pingpong balls, those would give us 2 points because its easier to fit in the bin. Etc.
; DIFFICULTY: 1

(define (game 616abb33ebe1d6112545f76d-38) (:domain medium-objects-room-v1)  ; 38

(:constraints (and 
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring maximize
    (* 5 (count-nonoverlapping throwToBin))
))

; 614fb15adc48d3f9ffcadd41-39 (many-objects-room-v1)
; SETUP: we need a ball
; GAMEPLAY: through the ball against the wall 
; SCORING: if the ball reflected back to the player, he/she score a point. 
; DIFFICULTY: 0
(define (game 614fb15adc48d3f9ffcadd41-39) (:domain many-objects-room-v1)  ; 39
(:constraints (and 
    (preference ballThrownToWallToAgent
        (exists (?b - ball ?w - wall) 
            (then
                (once (agent_holds ?b))
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?w ?b)
                )
                (once (or (agent_holds ?b) (touch agent ?b)))
            )
        )
    )
))
(:scoring maximize
    (count-nonoverlapping ballThrownToWallToAgent)
))

; 5c71bdec87f8cd0001b458f5-40 (many-objects-room-v1)
; SETUP: Set up the ramp on one end of the rug
; GAMEPLAY: use the beach ball and send it down the ramp 
; SCORING: The balls that land in pink are worth 1, the balls that land in orange and blue are worth 3, the balls that land in yellow are worth 2, the balls that land in purple are worth 4. The balls that land in white are worth -1, the balls that fall off the board are worth 0
; DIFFICULTY: 1
(define (game 5c71bdec87f8cd0001b458f5-40) (:domain many-objects-room-v1)  ; 40
(:setup (and 
    (exists (?r - curved_wooden_ramp) (game-conserved (adjacent ?r rug)))
))
(:constraints (and 
    (forall (?c - color)
        (preference ballRolledOnRampToRug
            (exists (?b - beachball ?r - curved_wooden_ramp)
                (then 
                    (once (agent_holds ?b))
                    (hold-while 
                        (and (not (agent_holds ?b)) (in_motion ?b))
                        (on ?r ?b)    
                    ) 
                    (once (and (not (in_motion ?b)) (on rug ?b) (rug_color_under ?b ?c)))
                )
            )
        )
    )
))
(:scoring maximize (+
    (count-nonoverlapping ballRolledOnRampToRug:pink)
    (* 2 (count-nonoverlapping ballRolledOnRampToRug:yellow))
    (* 3 (count-nonoverlapping ballRolledOnRampToRug:orange))
    (* 3 (count-nonoverlapping ballRolledOnRampToRug:green))
    (* 4 (count-nonoverlapping ballRolledOnRampToRug:purple))
    (* (- 1) (count-nonoverlapping ballRolledOnRampToRug:white))
)))

; 5f8d77f0b348950659f1919e-41 (many-objects-room-v1)
; SETUP: All items have been moved into one side of the room . A line on the floor has been drawn by using bridge blocks that have been frozen
; GAMEPLAY: The object of the game is to get the items from one side of the room as fast as possible.
; SCORING: You get 30 seconds and you get 1 point for each item that you move into the other half of the room in that time.
; You must count the seconds out loud or use a timer if you have one
; DIFFICULTY: 2
(define (game 5f8d77f0b348950659f1919e-41) (:domain many-objects-room-v1)  ; 41
(:setup (and 
    (exists (?w1 ?w2 - wall) (and  
        (game-conserved (opposite ?w1 ?w2))
        (forall (?b - bridge_block) (game-conserved (and 
            (on floor ?b)
            (= (distance ?w1 ?b) (distance ?w2 ?b))    
        )))
        (forall (?g - game_object) (game-optional (or 
            (type ?g bridge_block)
            (> (distance ?w1 ?g) (distance ?w2 ?g))
        )))
    ))
))
(:constraints (and
    (forall (?w1 ?w2 - wall)  
        (preference objectMovedRoomSide (exists (?g - game_object) 
            (then
                (once (and 
                    (not (agent_holds ?g))
                    (not (in_motion ?g))
                    (not (type ?g bridge_block))
                    (> (distance ?w1 ?b) (distance ?w2 ?b))
                ))
                (hold (or 
                    (agent_holds ?g)
                    (in_motion ?g)
                ))
                (once (and 
                    (not (in_motion ?g))
                    (< (distance ?w1 ?b) (distance ?w2 ?b))
                ))
            )
        ))
    )
))
(:terminal 
    (>= (total-time) 30)
)
(:scoring maximize
    (count-maximal-once-per-objects objectMovedRoomSide)
))

; 5edc195a95d5090e1c3f91b2-42 (few-objects-room-v1)
; SETUP: Move the bin to a location reasonably away from other objects. Move the dodgeballs to the ground a few "meters" away from the bin. Crouch.
; GAMEPLAY: Can play a basketball like game where you shoot the dodgeball into the bin. Crouch and use left click the interact with the object. Next, hold the ball and aim towards the bin. Pick up the ball from where it lands and go back to the original position to play again.
; SCORING: You get 5 chances to shoot the dodgeball into the bin from the same location. Every time it lands successfully into the bin, you get one point.
; DIFFICULTY: 2
(define (game 5edc195a95d5090e1c3f91b-42) (:domain few-objects-room-v1)  ; 42
(:setup (and 
    (exists (?h - hexagonal_bin) (and 
        (forall (?g - game_object) (game-optional (or
            (= ?h ?g)
            (> (distance ?h ?g) 1) 
        )))      
        (forall (?d - dodgeball) (game-optional (and
            (> (distance ?h ?d) 2) 
            (< (distance ?h ?d) 6) 
        )))
    ))
))
(:constraints (and 
    (preference throwBallFromOtherBallToBin 
        (exists (?d1 ?d2 - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d1) (adjacent agent ?d2)))
                (hold (and (not (agent_holds ?d1)) (in_motion ?d1))) 
                (once (and (not (in_motion ?d1)) (in ?h ?d1)))
            )
        )
    )
    (preference throwAttempt
        (exists (?d - dodgeball)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:terminal 
    (>= (count-nonoverlapping throwAttempt) 5)
)
(:scoring maximize
    (count-same-positions throwBallFromOtherBallToBin)
))

; 617378aeffbba11d8971051c-43 (medium-objects-room-v1)
; SETUP: Pick up the dog bed and place it in the center of the room. Collect the three balls.
; GAMEPLAY: Throw each of the balls and try to get them to land in the dog bed.
; SCORING: 1 point if you get the basketball to land in the dog bed. 2 points for the beach ball and 3 points for the dodgeball. 1 extra bonus point if you first bounce one of the balls off a wall.
; DIFFICULTY: 2

(define (game 617378aeffbba11d8971051c-43) (:domain medium-objects-room-v1)  ; 43
(:setup (and 
    (exists (?d - doggie_bed) (game-conserved (< (distance room_center ?d) 1)))
))
(:constraints (and 
    (forall (?b - ball) (and 
        (preference throwBallToBin
            (exists (?d - doggie_bed)
                (then 
                    (once (agent_holds ?b))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b) (not (exists (?w - wall) (touch ?w ?b ))))) 
                    (once (and (not (in_motion ?b)) (on ?d ?b)))
                )
            )
        )
        (preference throwBallToBinOffWall
            (exists (?d - doggie_bed ?w - wall)
                (then 
                    (once (agent_holds ?b))
                    (hold-while 
                        (and (not (agent_holds ?d)) (in_motion ?b))
                        (touch ?w ?b)    
                    ) 
                    (once (and (not (in_motion ?b)) (on ?d ?b)))
                )
            )
        )  
    ))
))
(:scoring maximize (+
    (count-nonoverlapping throwBallToBin:basketball)
    (* 2 (count-nonoverlapping throwBallToBin:beachball))
    (* 3 (count-nonoverlapping throwBallToBin:dodgeball))
    (* 2 (count-nonoverlapping throwBallToBinOffWall:basketball))
    (* 3 (count-nonoverlapping throwBallToBinOffWall:beachball))
    (* 4 (count-nonoverlapping throwBallToBinOffWall:dodgeball))
)))

; 44 is another find the hidden object game

; 60e7044ddc2523fab6cbc0cd-45 (many-objects-room-v1)
; SETUP: Grab a bear off the bed and place it in front of the bed on the floor right in the middle. Move the pillow in the middle of the bed somewhere where it won't get in the way. Grab the second bear and put it behind the first, sitting on the bed. 
; GAMEPLAY: Hit the bears! Stand next to the sliding glass door near the computer desk.  Use the three dodge balls and three golf balls to try and knock over the bears. If you run out of balls before knocking over the bears, you get zero points. 
; SCORING: Using a dodge ball gets you 1 point if you knock over a bear. Using a golf ball gives you 2 points if you knock over a bear. 
; DIFFICULTY: 0
(define (game 60e7044ddc2523fab6cbc0cd-45) (:domain many-objects-room-v1)  ; 45
(:setup (and 
    (exists (?t1 ?t2 - teddy_bear) (game-optional (and 
        (on floor ?t1)
        (on bed ?t2)
        ; TODO: is the below nicer than (= (z_position ?t1) (z_position ?T2))
        (equal_z_position ?t1 ?t2)
        (equal_z_position ?t1 bed)
    )))
))
(:constraints (and 
    (forall (?b - (either golfball dodgeball)) (and 
        (preference throwKnocksOverBear (exists (?t - teddy_bear ?s - sliding_door) 
            (then
                (once (and 
                    (agent_holds ?b)
                    (adjacent agent desk)
                    (adjacent agent ?s)
                    (equal_z_position ?t bed)
                    ; (= (z_position ?t) (z_position bed))
                ))
                (hold-while
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (touch ?b ?t)
                )
                (once (in_motion ?t))
            )
        ))
        (preference throwAttempt (exists (?s - sliding_door)
            (then
                (once (and (agent_holds ?b) (adjacent agent desk) (adjacent agent ?s)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        ))
    ))
))
(:terminal (or 
    (> (count-maximal-nonoverlapping throwAttempt) 1)
    (>= (count-once-per-objects throwAttempt) 6)
))
(:scoring maximize (+
    (count-once-per-objects throwKnocksOverBear:dodgeball)
    (* 2 (count-once-per-objects throwKnocksOverBear:golfball))   
)))

; 5d5b0dd7c032a2001ad7cf5d-46 (few-objects-room-v1)
; SETUP: Take the curved ramp and place it somewhere in the middle of the room, with the far end of it being the higher end. Then find the purple dodgeball in one corner of the room, and pick it up.
; GAMEPLAY: With the curved ramp set up in the middle of the room, stand several feet from it with the purple ball in your hands. You will throw the purple dodgeball at the curved ramped, with the goal of trying to get it to land on the bed. 
; SCORING: For each instance that you throw the purple dodgeball at the curved ramp, and it lands on the bed, you gain 1 point. For each instance it lands on the ground somewhere, you gain zero points. Finally, for each instance the dodgeball richochets backwards and instead hits you, you lose 1 point.
; DIFFICULTY: 3
(define (game 5d5b0dd7c032a2001ad7cf5d-46) (:domain few-objects-room-v1)  ; 46
(:setup (and 
    (exists (?c - curved_wooden_ramp) (game-conserved
        (< (distance ?c room_center) 3)  
    ))
))
(:constraints (and 
    (preference ballThrownToRampToBed (exists (?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds pink_dodgeball) (faces agent ?c)))
            (hold-while
                (and (in_motion pink_dodgeball) (not (agent_holds pink_dodgeball)))
                (touch pink_dodgeball ?c)
            )
            (once (and (not (in_motion pink_dodgeball)) (on bed pink_dodgeball)))
        )
    ))
    (preference ballThrownHitsAgent (exists (?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds pink_dodgeball) (faces agent ?c)))
            (hold-while
                (and (in_motion pink_dodgeball) (not (agent_holds pink_dodgeball)))
                (touch pink_dodgeball ?c)
            )
            (once (and (touch pink_dodgeball agent) (not (agent_holds pink_dodgeball))))
        )
    ))
))
(:scoring maximize (+ 
    (count-nonoverlapping ballThrownToRampToBed)
    (* (- 1) (count-nonoverlapping ballThrownHitsAgent))
)))

; 5d470786da637a00014ba26f-47 (many-objects-room-v1)
; SETUP: None, you just need to pick up the Beachball.
; GAMEPLAY: Standing off the multi-coloured rug opposite the green small ramp you can charge a throw and rebound the beachball off the small green ramp and allow it to roll back towards you and then score points based on how far it rolls and which coloured section of the rug it stops in.
; SCORING: The first section in red could be 1 point for a poor throw or bounce, the larger pink section can be 3 points but the incentive should be on trying to roll it back the furthest to land in the green section for 10 points without it rolling off the mat which is possible with a fully charged shot, if you roll off the rug then you would score 0 points.
; DIFFICULTY: 1
(define (game 5d470786da637a00014ba26f-47) (:domain many-objects-room-v1)  ; 47
(:constraints (and 
    (forall (?c - color) 
        (preference beachballBouncedOffRamp
            (exists (?b - beachball ?r - green_triangular_ramp)
                (then
                    (once (and (agent_holds ?b) (not (on rug agent))))
                    (hold-while
                        (and (in_motion ?b) (not (agent_holds ?b)))
                        (touch ?b ?r)
                    )
                    (once (and (not (in_motion ?b)) (on rug ?b) (rug_color_under ?b ?c)))
                )
            )
        )
    )
))
(:scoring maximize (+ 
    (count-nonoverlapping beachballBouncedOffRamp:red)
    (* 3 (count-nonoverlapping beachballBouncedOffRamp:pink))
    (* 10 (count-nonoverlapping beachballBouncedOffRamp:pink))
)))

; TODO: this is a crude approximation of 48 -- let's hope it's reasonable?

; 61254c5a6facc8ed023a64de-48 (medium-objects-room-v1)
; SETUP: The game player will play is basketball. At the beginning of the game you will have to build a basketball basket on the center of room. You will have few minutes to do it. You can use various items in the room like cylinderblocks to make a pipe and use a bin as a basket. But be careful, you should also take care of the safety of play (not break something).
; The bin should be stable on top of the structure you have build. Player will have an extra abilities to freeze (except bin, it will be stable as long as the construction is stable) up to 3 different items. Every other usage of it will cost him 15 points.
; Your goal is to achieve at least 200 points with 50 ball throwings.
; Basket should be built finally at eye level when you are not crouching.
; GAMEPLAY: Game will be basketball, player will need to throw 3 different sized balls to it - beachball, dodgeball and basketball. They are different sized and different pointed.
; Then every time he throws a ball player will be spawned in random place in the room and player will need to hit the ball into a basket.
; SCORING: Firstly, player will gain extra 10 points if he use pillow or dog dogbed to secure desktop screens or notebook. He loose 20 points for any extra "freeze".
; Player needs to throw a ball to the bin without break built construction or something behind it (for example computers).
; He may use any ball you want, but:
; - if player score with smallest (dodge) ball you will earn 5 points;
; - if player score with basketball - you will earn 7 points;
; - if player score using beachball (which is too big for bin and it will be on it) - you will earn 15 points
; Extra points can be achieve if the player:
; - use pillows/dogbed/teddy bear to secure notebook (+10 points)
; - use pillow/dogbed/teddy bear to secure dekstop (+10 points)
; - use blinds to secure windows (2x 10 points)
; - secure door windows (+50 points - it's diffifult to secure it), example: https://i.imgur.com/hH6ClWQ.png
; - hide alarmclock in drawer (10 points)
; - hide cellphone in drawer (10 points)
; The fewer objects change its default position to build a basket and put it in the position of the eyes, the better.
; Every time a item will be moved (except balls, opening and closing drawer and move blinds) a player will loose 5 points. That's why player gains extra points if he remember to secure room and he make the least amount of movements.
; If the player broke window, notebook or 
; DIFFICULTY: 4
(define (game 61254c5a6facc8ed023a64de-48) (:domain medium-objects-room-v1)  ; 48
(:setup (and 
    (exists (?b - building ?h - hexagonal_bin) (game-conserved (and 
        (in ?b ?h)
        (>= (building_size ?b) 4) ; TODO: could also quantify out additional objects
        (not (exists (?g - game_object) (and (in ?b ?g) (on ?h ?g))))
        (< (distance ?b room_center) 1)
    )))
))
(:constraints (and 
    (forall (?d - (either dodgeball basketball beachball))
        (preference ballThrownToBin (exists (?b - building ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (or (in ?h ?d) (on ?h ?d))))
            )
        ))
    )
    (preference itemsHidingScreens 
        (exists (?s - (either desktop laptop) ?o - (either pillow doggie_bed teddy_bear)) 
            (at-end (on ?s ?o))    
        )
    )
    (preference objectsHidden
        (exists (?o - (either alarm_clock cellphone) ?d - drawer)
            (at-end (in ?d ?o))
        )
    )
    (preference blindsOpened
        (exists (?b - blinds)
            (at-end (open ?b))  ; blinds being open = they were pulled down
        )
    )
    (preference objectMoved
        (exists (?g - game_object) 
            (then
                (once (and 
                    (not (in_motion ?g)) 
                    (not (type ?g ball))
                    (not (type ?g drawer))
                    (not (type ?g blinds))
                ))
                (hold (in_motion ?g))
                (once (not (in_motion ?g)))
            )
        )
    )
))
(:scoring maximize (+ 
    (* 5 (count-nonoverlapping ballThrownToBin:dodgeball))
    (* 7 (count-nonoverlapping ballThrownToBin:basketball))
    (* 15 (count-nonoverlapping ballThrownToBin:beachball))
    (* 10 (count-once-per-objects itemsHidingScreens))
    (* 10 (count-once-per-objects objectsHidden))
    (* 10 (count-once-per-objects blindsOpened))
    (* (- 5) (count-nonoverlapping objectMoved))
)))

; 60ddfb3db6a71ad9ba75e387-49 (many-objects-room-v1)
; SETUP: The first game: The basketball game. You need to stand at the door on the green little ball, this is your starting point. 
; GAMEPLAY: On your right you there should be 3 balls, your task is to grab these balls and try to throw in into the (green) bin that is between the two windows. becareful, you only have one try with one ball!
; SCORING: Each successful throw represent 10 points.
; DIFFICULTY: 3
(define (game 60ddfb3db6a71ad9ba75e387-49) (:domain many-objects-room-v1)  ; 49
(:setup (and 
    (game-conserved (< (distance green_golfball door) 0.5))
    (forall (?d - dodgeball) (game-optional (< (distance green_golfball ?d) 1)))
))
(:constraints (and 
    (forall (?d - dodgeball) (and 
        (preference dodgeballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (and 
                    (adjacent agent green_golfball)
                    (adjacent agent door)
                    (agent_holds ?d)
                ))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
        (preference throwAttemptFromDoor 
            (then
                (once (and 
                    (adjacent agent green_golfball)
                    (adjacent agent door)
                    (agent_holds ?d)
                ))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (not (in_motion ?d)))
            )
        )
    ))
))
(:terminal (or 
    (> (count-maximal-nonoverlapping throwAttemptFromDoor) 1)
    (>= (count-once-per-objects throwAttemptFromDoor) 3)
))
(:scoring maximize
    (* 10 (count-once-per-objects dodgeballThrownToBin))
))

; 5f3aee04e30eac7cb73b416e-50 (medium-objects-room-v1)
; SETUP: Dumpster must be on the middle.
; GAMEPLAY: In this game, you have to throw as many items to the dumpster, as it's possible.
; SCORING: Every item in the dumpster equal one point.
; DIFFICULTY: 1
(define (game 5f3aee04e30eac7cb73b416e-50) (:domain medium-objects-room-v1)  ; 50
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (< (distance room_center ?h) 1)))
))
(:constraints (and 
    (preference gameObjectToBin (exists (?g - game_object ?h - hexagonal_bin)
        (then 
            (once (not (agent_holds ?g)))
            (hold (or (agent_holds ?g) (in_motion ?g)))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
(:scoring maximize
    (count-once-per-objects gameObjectToBin)
))

; 5ff4a242cbe069bc27d9278b-51 (few-objects-room-v1)
; SETUP: Take some dodgeball
; GAMEPLAY: throw the dodgeball in the bin
; SCORING: If the dodgeball enter inside the bin you got one point
; DIFFICULTY: 1

(define (game 5ff4a242cbe069bc27d9278b-51) (:domain few-objects-room-v1)  ; 51
(:constraints (and 
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring maximize
    (count-nonoverlapping throwToBin)  
))

; 602d84f17cdd707e9caed37a-52 (few-objects-room-v1)
; SETUP: Stand on the carpet within the reach of all 6 Cube Blocks.
; GAMEPLAY: The person stand/crouch on the carpet and throw cube blocks on the desk. 
; The goal is to put blocks on the desk and not to destroy/move the computer, lamp or laptop. 
; Player is not allowed to leave the rug.
; There are 6 CubeBlocks, so the player can score up to 6 points.
; SCORING: 1 CubeBlock on the desk = +1 point
; Moving laptop, lamp or monitor / leaving the rug = 0 points
; DIFFICULTY: 1
(define (game 602d84f17cdd707e9caed37a-52) (:domain few-objects-room-v1)  ; 52
(:constraints (and 
    (preference blockFromRugToDesk (exists (?c - cube_block ) 
        (then 
            (once (and (on rug agent) (agent_holds ?c)))
            (hold (and 
                (on rug agent)
                (in_motion ?c)
                (not (agent_holds ?c))
                (not (exists (?o - (either lamp desktop laptop)) (or (broken ?o) (in_motion ?o))))
            ))
            (once (and (on rug agent) (on desk ?c) (not (in_motion ?c))))
        )
    ))
))
(:scoring maximize
    (count-once-per-objects blockFromRugToDesk)
))

; 5f0cc31363e0816c1b0db7e1-53 (few-objects-room-v1)
; SETUP: To prepare room for the game, Player have to clean the room and to move some things to the right places. 
; GAMEPLAY: To play my game Player have to put all cubes to the shelves, to put dodgeballs to the bin, and move the bin to the corner. Also to hide small things. 
; SCORING: The more  quick and clear movement, than Player will have higher score. Each right movement will bring 5 points. 
; DIFFICULTY: 2
(define (game 5f0cc31363e0816c1b0db7e1-53) (:domain few-objects-room-v1)  ; 53
(:constraints (and 
    (preference dodgeballsInPlace 
        (exists (?d - dodgeball ?h - hexagonal_bin ?w1 ?w2 - wall)
            (at-end (and (in ?h ?d) (adjacent ?h ?w1) (adjacent ?h ?w2)))
        )
    )
    (preference blocksInPlace
        (exists (?c - cube_block ?s - shelf)
            (at-end (on ?s ?c))
        )
    )
    (preference smallItemsInPlace
        (exists (?o - (either cellphone key_chain mug credit_card cd watch alarm_clock) ?d - drawer)
            (at-end (and 
                (in ?d ?o)
            ))
        )
    )
))
(:scoring maximize (+ 
    (* 5 (count-once-per-objects dodgeballsInPlace))
    (* 5 (count-once-per-objects blocksInPlace))
    (* 5 (count-once-per-objects smallItemsInPlace))
)))

; 61541833a06877a656163b10-54 (few-objects-room-v1)
; SETUP: No need for changes
; GAMEPLAY: You could play stacking cube blocks. In order to do so, you are required any cube block that you want to stack
; SCORING: Each cube block that you moved and successfully stacked will be counted as one point. You can only pick up any block 2 times. So if it falls only a second chance. As long as the tower stamd for 2 seconds. The highest state reached at any point in time will be counted
; DIFFICULTY: 2
(define (game 61541833a06877a656163b10-54) (:domain few-objects-room-v1)  ; 54
(:constraints (and 
    (forall (?b - building) 
        (preference blockPlacedInBuilding (exists (?l - cube_block)
            (then
                (once (agent_holds ?l))
                (hold (and (in_motion ?l) (not (agent_holds ?l))))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
    )
    (forall (?l - cube_block) 
        (preference blockPickedUp 
            (then
                (once (not (agent_holds ?l)))
                (hold (agent_holds ?l))
                (once (not (agent_holds ?l)))
            )
        )
    )
))
(:terminal
    (>= (count-maximal-nonoverlapping blockPickedUp) 3)
)
(:scoring maximize (+
    (count-maximal-overlapping blockPlacedInBuilding)
)))

; 5f7654f879a4420e6d20971b-55 (few-objects-room-v1)
; SETUP: Find the trash bin and move it to the center of the room (across from the mirror). 
; GAMEPLAY: To play my game, players must find as items as possible, pick them up, and throw them in the bin.
; SCORING: To win a point, players must successfully land an item inside the bin on the first throw. 
; DIFFICULTY: 2
(define (game 5f7654f879a4420e6d20971b-55) (:domain few-objects-room-v1)  ; 55
(:setup (and 
    (exists (?h - hexagonal_bin)
        (game-conserved (< (distance ?h room_center) 1))
    )
))
(:constraints (and 
    (preference objectToBinOnFirstTry (exists (?o - game_object ?h - hexagonal_bin)
        (then 
            (once (game_start))
            (hold (not (agent_holds ?o)))
            (hold (agent_holds ?o))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (in ?h ?o)))
            (hold (not (agent_holds ?o)))
        )
    ))
))
(:scoring maximize
    (count-once-per-objects objectToBinOnFirstTry)
))

; 604a7e9f84bf0e7937200df5-56 (few-objects-room-v1)
; SETUP: grab a dodgeball from the drawers above the bed. go as far away from the garbage bin as you can and throw the dodgeball into the bin
; GAMEPLAY: you have to make the ball into the bin by throwing it
; SCORING: you have 3 tries to make it. the less tries the more points. the cleaner you sink it the more points,
; DIFFICULTY: 4
(define (game 604a7e9f84bf0e7937200df5-56) (:domain few-objects-room-v1)  ; 56
(:constraints (and 
    ; TODO: are we okay with ignoring the subjectivity?
    ; "you have 3 tries to make it. the less tries the more points. the cleaner you sink it the more points""
    (preference throwFromDoorToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then 
            (once (and (agent_holds ?d) (adjacent agent door)))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
    (preference throwAttempt (exists (?d - dodgeball)
        (then 
            (once (agent_holds ?d))
            (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
            (once (not (in_motion ?d)))
        )
    ))
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 3)
)
(:scoring maximize
    (count-nonoverlapping throwFromDoorToBin)
))

; 61623853a4ccad551beeb11a-57 (medium-objects-room-v1)
; SETUP: the objects must stay in in the inicial place as 1st shown
; GAMEPLAY: The game is to tidy up the room. The objects should be placed as follows:
; - book on the shelf of the desk;
; - pensil, pen and cd on the other shelf of the desk;
; - dodge and basketball in the bin;
; - beach ball on the carpet;
; - Cell phone, keys and credit card on the bedside table's drawers;
; - Watch on the shelf.
; SCORING: each time you fix an object that must be fixed, you earn 5 points
; DIFFICULTY: 2
(define (game 61623853a4ccad551beeb11a-57) (:domain medium-objects-room-v1)  ; 57
(:constraints (and 
    (preference bookOnDeskShelf (exists (?b - book ?d - desk_shelf)
        (at-end (and 
            (on ?d ?b)
            (not (exists (?o - (either pencil pen cd)) (on ?d ?o)))
        ))
    ))
    (preference otherObjectsOnDeskShelf (exists (?o - (either pencil pen cd) ?d - desk_shelf)
        (at-end (and 
            (on ?d ?o)
            (not (exists (?b - book) (on ?d ?b)))
        ))
    ))
    (preference dodgeballAndBasketballInBin (exists (?b - (either dodgeball basketball) ?h - hexagonal_bin)
        (at-end (in ?h ?b))
    ))
    (preference beachballOnRug (exists (?b - beachball ?r - rug)
        (at-end (on ?r ?b))
    ))
    (preference smallItemsInPlace (exists (?o - (either cellphone key_chain cd) ?d - drawer)
        (at-end (in ?d ?o))
    ))
    (preference watchOnShelf (exists (?w - watch ?s - shelf)
        (at-end (on ?s ?w))
    ))
))
(:scoring maximize (+ 
    (count-once-per-objects bookOnDeskShelf)
    (count-once-per-objects otherObjectsOnDeskShelf)
    (count-once-per-objects dodgeballAndBasketballInBin)    
    (count-once-per-objects beachballOnRug)
    (count-once-per-objects smallItemsInPlace)
    (count-once-per-objects watchOnShelf)
)))

; 5f0a5a99dbbf721316f118e2-58 (medium-objects-room-v1)
; SETUP: Using the blocks on the shelf, use the short cylinder, long cylinder, pyramid, cube, flat rectangle and the bridge blocks to make a tower. Hide the other set of identical blocks around the room.
; GAMEPLAY: The main objective is to make a second block structure that is a match to the first one. However, the blocks you need to build the structure are missing. They are hidden around the room and you need to find them. 
; SCORING: First, you need to find the hidden blocks. There are six blocks total.  They are worth 5 points, meaning the highest earning is 30 points.
; Next, you use the blocks to match the first tower. Before you start building, you gain 100 more points. Then, 10 points get taken off of your total score every time the tower falls over or it doesn't match.
; DIFFICULTY: 3    
(define (game 5f0a5a99dbbf721316f118e2-58) (:domain medium-objects-room-v1)  ; 58
(:setup (and 
    (exists (?b - building) (and 
        (game-conserved (= (building_size ?b) 6))
        (forall (?l - block) (or 
            (game-conserved (and 
                    (in ?b ?l) 
                    (not (exists (?l2 - block) (and 
                        (in ?b ?l2)
                        (not (= ?l ?l2))
                        (same_type ?l ?l2)
                    )))
            ))
            (game-optional (not (exists (?s - shelf) (on ?s ?l))))
        ))
    ))        
))
(:constraints (and 
    (preference gameBlockFound (exists (?l - block)
        (then 
            (once (game_start))
            (hold (not (exists (?b - building) (and (in ?b ?l) (is_setup_object ?b)))))
            (once (agent_holds ?l))
        )
    ))
    (preference towerFallsWhileBuilding (exists (?b - building ?l1 ?l2 - block)
        (then
            (once (and (in ?b ?l1) (agent_holds ?l2) (not (is_setup_object ?b))))
            (hold-while 
                (and
                    (not (agent_holds ?l1)) 
                    (in ?b ?l1)
                    (or 
                        (agent_holds ?l2) 
                        (and (not (agent_holds ?l2)) (in_motion ?l2))
                    )
                )
                (touch ?l1 ?l2)
            )
            (hold (and 
                (in_motion ?l1)
                (not (agent_holds ?l1))
            ))
            (once (not (in_motion ?l1)))
        )
    ))
    (preference matchingBuildingBuilt (exists (?b1 ?b2 - building)
        (at-end (and 
            (is_setup_object ?b1) 
            (not (is_setup_object ?b2))
            (forall (?l1 ?l2 - block) (or 
                (not (in ?b1 ?l1))
                (not (in ?b1 ?l2))
                (not (on ?l1 ?l2))
                (exists (?l3 ?l4 - block) (and 
                    (in ?b2 ?l3)
                    (in ?b2 ?l4)
                    (on ?l3 ?l4)
                    (same_type ?l1 ?l3)
                    (same_type ?l2 ?l4)
                ))
            ))
        ))
    ))
))
(:scoring maximize (+ 
    (* 5 (count-once-per-objects gameBlockFound))
    (* 100 (count-once matchingBuildingBuilt))
    (* (-10) (count-nonoverlapping towerFallsWhileBuilding))
)))

; 602a1735bf92e79a5e7cb632-59 (many-objects-room-v1)
; SETUP: Move the bin near the door and get any of the balls (golf, dogeball or even basketball)
; GAMEPLAY: Throw the differents balls in the bin (golf, dogeball and basketball)
; SCORING: Throwing the differents ball in the bin give these points
; - Golf balls : 2 points
; - Dogeballs : 3 points
; - Basketball : 4 points
; DIFFICULTY: 3
(define (game 602a1735bf92e79a5e7cb632-59) (:domain many-objects-room-v1)  ; 59
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (< (distance ?h door) 1)))
))
(:constraints (and 
    (forall (?b - (either golfball dodgeball beachball))
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
(:scoring maximize (+ 
    (* 2 (count-nonoverlapping ballThrownToBin:golfball))
    (* 3 (count-nonoverlapping ballThrownToBin:dodgeball))
    (* 4 (count-nonoverlapping ballThrownToBin:beachball))
)))

; 60 is invalid

; 6086efbd71dc51bb8d6a1a5e-61 (many-objects-room-v1)
; SETUP: The setum of the room requires to drop a piece of flat rectblock for 1 game into the carpet and the 3 pyramidblocks at a certain distance. 
; GAMEPLAY: 1 game is to throw the 3 dodgeballs into the bin. Also make the longest tower using the cubeblocks on top of the recblock.
; SCORING: For the dodgeballs it is necessary to drop them into the bin at the distance of the yellow reblock (10 points),  red pyramidblock (25 points) and blue pyramidblock (50 points)
; if it is too close to the bin, no points will be given (0). Adittional points will be given to the farthest distance to the bin (all 3 balls in the blue pyramid for 100 additional points). For the tower, there are 30 points for each recblock succesfully placed in a vertical position, and 100 points in total for making the tower of 3.
; DIFFICULTY: 3
(define (game 6086efbd71dc51bb8d6a1a5e-61) (:domain many-objects-room-v1)  ; 61
(:setup (game-conserved (and 
    (exists (?f - flat_block) (on rug ?f))
    (forall (?p - pyramid_block) (on floor ?p))
    (exists (?p1 - yellow_pyramid_block ?p2 - red_pyramid_block ?p3 - blue_pyramid_block ?h - hexagonal_bin) 
        (and 
            (> (distance ?h ?p2) (distance ?h ?p1)) 
            (> (distance ?h ?p3) (distance ?h ?p2))    
        )
    )
)))
(:constraints (and 
    (forall (?p - pyramid_block)
        (preference dodgeballFromBlockToBin (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (adjacent agent ?p)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))        
            )  
        ))
    )
    (preference cubeBlockInBuilding (exists (?b - building ?l - cube_block ?f - flat_block) 
        (at-end (and 
              (is_setup_object ?f)
              (in ?b ?f)
              (in ?b ?l)
        ))
    ))
))
(:scoring maximize (+ 
    (* 10 (count-nonoverlapping dodgeballFromBlockToBin:yellow_pyramid_block))
    (* 25 (count-nonoverlapping dodgeballFromBlockToBin:red_pyramid_block))
    (* 50 (count-nonoverlapping dodgeballFromBlockToBin:blue_pyramid_block))
    (* 100 (= (count-once-per-objects dodgeballFromBlockToBin:blue_pyramid_block) 3))
    (* 10 (count-once-per-objects cubeBlockInBuilding))
    (* 100 (= (count-once-per-objects cubeBlockInBuilding) 3))
)))

; 601c84e07ab4907ded068d0d-62 (medium-objects-room-v1)
; SETUP: All items should be in their place of origin (laptop on desk etc.).
; GAMEPLAY: A player must collect 2 different items and throw them from one end of the room on the bed. If any item fall of the bed a player should start a game again.
; SCORING: Big objects like laptop, mattress, chair give a player additional 5 points. Every lost round (when any item fall off the bed) cause minus 5 points. 1 small/middle item gives a player 1 point.
; DIFFICULTY: 3
(define (game 601c84e07ab4907ded068d0d-62) (:domain medium-objects-room-v1)  ; 62
(:constraints (and 
    (preference bigObjectThrownToBed (exists (?o - (either chair laptop doggie_bed))
        (then
            (once (and (agent_holds ?o) (adjacent agent desk)))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (on bed ?o)))
        )
    ))
    (preference smallObjectThrownToBed (exists (?o - game_object)
        (then
            (once (and 
                (agent_holds ?o) 
                (adjacent agent desk) 
                (not (exists (?o2 - (either chair laptop doggie_bed)) (= ?o ?o2)))
            ))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (on bed ?o)))
        )
    ))
    (preference failedThrowAttempt (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (adjacent agent desk)))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (not (on bed ?o))))
        )
    ))
))
(:scoring maximize (+
    (count-nonoverlapping smallObjectThrownToBed)
    (* 5 (count-nonoverlapping bigObjectThrownToBed))
    (* (- 5) (count-nonoverlapping failedThrowAttempt))
)))


; 60bb3b463887c2f9d1385cce-63 (medium-objects-room-v1)
; SETUP: Ensure there are no objects in the middle of the room.
; GAMEPLAY: The game is called Object Stack. Using the objects in the room, the player is supposed to stack objects vertically, on top of each other.
; For increased points use any non-block objects.
; The game ends when an object falls from the stack.
; SCORING: The scoring system is 1 point for each wooden block (available on the shelf in the middle of the room) stacked and 2 points for each non-block object (everything else) stacked.
; DIFFICULTY: 2
(define (game 60bb3b463887c2f9d1385cce-63) (:domain medium-objects-room-v1)  ; 63
(:constraints (and 
    (preference towerFallsWhileBuilding (exists (?b - building ?l1 ?l2 - block)
        (then
            (once (and (in ?b ?l1) (agent_holds ?l2) (not (is_setup_object ?b))))
            (hold-while 
                (and
                    (not (agent_holds ?l1)) 
                    (in ?b ?l1)
                    (or 
                        (agent_holds ?l2) 
                        (and (not (agent_holds ?l2)) (in_motion ?l2))
                    )
                )
                (touch ?l1 ?l2)
            )
            (hold (and 
                (in_motion ?l1)
                (not (agent_holds ?l1))
            ))
            (once (not (in_motion ?l1)))
        )
    ))
    (forall (?b - building) (and 
        (preference blockPlacedInBuilding (exists (?l - block)
            (then
                (once (agent_holds ?l))
                (hold (and (in_motion ?l) (not (agent_holds ?l))))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
        (preference nonBlockPlacedInBuilding (exists (?o - game_object)
            (then
                (once (and (agent_holds ?o) (not (type ?o block))))
                (hold (and (in_motion ?l) (not (agent_holds ?l))))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
    ))
))
(:terminal
    (>= (count-once towerFallsWhileBuilding) 1)
)
(:scoring maximize (+ 
    (count-maximal-overlapping blockPlacedInBuilding)
    ( * 2 (count-maximal-overlapping nonBlockPlacedInBuilding))
)))

; 5aeb24e22bd17300018779f2-64 (many-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: Throwing the dodgeballs into the bin, standing at different distances.
; SCORING: The further you are, the more points: next to the bin = 1p, carpet edge = 2p, against the wall = 3p.
; DIFFICULTY: 1
(define (game 5aeb24e22bd17300018779f2-64) (:domain many-objects-room-v1)  ; 64
(:constraints (and 
    (forall (?o - (either hexagonal_bin rug wall))
        (preference ballThrownFromObjectToBin (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (adjacent agent ?o)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
    )
))
(:scoring maximize (+ 
    (count-nonoverlapping ballThrownFromObjectToBin:hexagonal_bin)   
    (* 2 (count-nonoverlapping ballThrownFromObjectToBin:rug))
    (* 3 (count-nonoverlapping ballThrownFromObjectToBin:wall))
)))

; 56cf6e8d31a5bc0006e1cdf5-65 (many-objects-room-v1)
; SETUP: enter
; GAMEPLAY: get all the balls on the bed
; SCORING: count balls after time x
; DIFFICULTY: 1
(define (game 56cf6e8d31a5bc0006e1cdf5-65) (:domain many-objects-room-v1)  ; 65
(:constraints (and 
    (preference ballOnBedAtEnd (exists (?b - ball)
        (at-end 
            (on bed ?b)
        )   
    ))
))
(:scoring maximize (count-once-per-objects ballOnBedAtEnd)
))

; 5f806f22e8159d0913945e35-66 (medium-objects-room-v1)
; SETUP: Take the two BridgeBlocks and the two CubeBlocks from the shelf, and place them on the ground near the door, not too close to each other (and freeze them).
; Take the two CylinderBlocks and put them next to the two LongCylinderBlocks on the lower shelf.
; Put aside all the other blocks somewhere, so that only the cylinders are on the (lower) shelf, the top one should be empty.
; Place the DogBed next to the SmallRamp on the ground, and take the Dodgeball.
; GAMEPLAY: The game is to guess where the Dodgeball will end up after throwing it.
; Pick one of the building blocks from the lower shelf and put it on the top shelf. The block with the corresponding color is the one you are betting on. (So you put a block on the top shelf to signify that you think that the block with the same color that's on the ground will be the one that's closest to the ball after the ball stops.) (We say that the light blue block has the same color as the dark blue one for the purpose of this game.)
; Stand (or crouch) on the DogBed with the Dodgeball in your hand, and throw it in the direction of the door (for extra challenge it could be a requirement that it has to bounce off of the door). Try to aim it so that it ends up next to the block you picked. After the ball stops, assess which block it is closest to (use your honest judgement, but if there's an equal distance to more blocks, throw it again).
; If you guessed correctly, the block you picked is removed from the shelves (permanently), but the corresponding block on the ground stays there! If you guessed incorrectly it should be put on the lower one again, and you can pick a block again (or leave it on the top one if that's your pick for the next round as well). When all 4 blocks are removed from the shelves, the game ends.
; SCORING: When you correctly guess the block, you get 10 points. For incorrect guesses you lose 1 point. (You can have negative amount of points as well.) When the last block is removed from the shelves, you get 100 points, just so you feel better about your performance, then the game ends. :)
; DIFFICULTY: 4 
(define (game 5f806f22e8159d0913945e35-66) (:domain medium-objects-room-v1)  ; 66
(:setup (and 
    (forall (?b - (either bridge_block cube_block)) 
        (game-conserved (< (distance ?b door) 1))    
    )
    (forall (?b - (either cylindrical_block tall_cylindrical_block)) 
        (game-optional (on bottom_shelf ?b))
    )
    (forall (?b - (either flat_block pyramid_block))
        (game-conserved (not (exists (?s - shelf) (on ?s ?b))))
    )
))
(:constraints (and 
    (forall (?b - (either cylindrical_block tall_cylindrical_block)) (and 
        (preference blockCorrectlyPicked (exists (?d - dodgeball ?o - doggie_bed ?tb - (either bridge_block cube_block))
            (then 
                (once (and 
                    (agent_holds ?d) 
                    (on agent ?o) 
                    (on top_shelf ?b)
                    (not (exists (?ob - block) 
                        (and 
                            (not (= ?b ?ob)) 
                            (on top_shelf ?ob)
                        )
                    ))
                ))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (agent_holds ?b))))
                (once (and 
                    (not (in_motion ?d)) 
                    (not (exists (?ob - block) (< (distance ?d ?ob) (distance ?d ?tb))))
                    (= (color ?b) (color ?tb))
                ))
            )
        ))
        (preference blockIncorrectlyPicked (exists (?d - dodgeball ?o - doggie_bed ?tb - (either bridge_block cube_block))
            (then 
                (once (and 
                    (agent_holds ?d) 
                    (on agent ?o) 
                    (on top_shelf ?b)
                    (not (exists (?ob - block) 
                        (and 
                            (not (= ?b ?ob)) 
                            (on top_shelf ?ob)
                        )
                    ))
                ))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (agent_holds ?b))))
                (once (and 
                    (not (in_motion ?d)) 
                    (not (exists (?ob - block) (< (distance ?d ?ob) (distance ?d ?tb))))
                    (not (= (color ?b) (color ?tb)))
                ))
            )
        ))
    ))   
))
(:terminal
    (>= (count-once-per-external-objects blockCorrectlyPicked) 4)
)
(:scoring maximize (+ 
    (* 10 (count-once-per-external-objects blockCorrectlyPicked))
    (* (- 1) (count-nonoverlapping blockIncorrectlyPicked))
    ( * 100 (>= (count-once-per-external-objects blockCorrectlyPicked) 4))
)))

; 60feca537ed1de34c8ddbbab-67 (medium-objects-room-v1)
; SETUP: Put 2 long cylinder blocks, 2 bridge blocks, 2 flat rect blocks and 4 cube blocks in a triangle formation like bowling pins, with the point of the triangle facing the bowler, and put them in front of the desk, remove the chairs to the side.
; GAMEPLAY: Use dodgeball, basketball or beachball as a bowling ball and hit the blocks, release the ball from on the carpet, cannot come out of the carpet.
; SCORING: If using dodgeball, the number of cubes that are knock down are the number of points a person gets; if using basketball, the number of points are the number of knocked down cubes multiplied by 0.7: if using beachball, the number of points are the number of knocked down cubes multiplied by 0.5. A single game consists of 8 frames, with each frame consisting of 2 chances to knock down for each person.
; DIFFICULTY: 1
(define (game 60feca537ed1de34c8ddbbab-67) (:domain medium-objects-room-v1)  ; 67
(:setup (and 
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 - (either tall_cylindrical_block bridge_block flat_block cube_block cylindrical_block))
        (game-optional (and 
            (= (distance desk ?b1) (distance desk ?b2) (distance desk ?b3) (distance desk ?b4))
            (= (distance desk ?b5) (distance desk ?b6) (distance desk ?b7))
            (= (distance desk ?b8) (distance desk ?b9))
            (< (distance desk ?b10) 2)
            (< (distance desk ?b1) (distance desk ?b5))
            (< (distance desk ?b5) (distance desk ?b8))
            (< (distance desk ?b8) (distance desk ?b10))
        ))
    )
    (forall (?c - chair) (game-conserved (not (adjacent_side desk front ?c))))
))
(:constraints (and 
    (forall (?b - ball) (and
        (preference ballKnocksBlockFromRug (exists (?l - block)
            (then 
                (once (and (agent_holds ?b) (on rug agent)))
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?l)
                    (in_motion ?l)
                )
            )
        ))
        (preference throwAttempt 
            (then 
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b))))
            )
        )
    ))
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 16)
)
(:scoring maximize (+ 
    (count-once-per-objects ballKnocksBlockFromRug:dodgeball)
    (* 0.7 (count-once-per-objects ballKnocksBlockFromRug:basketball))
    (* 0.5 (count-once-per-objects ballKnocksBlockFromRug:beachball))
)))

; 68 has subjective scoring -- I could attempt to objectify, but it's hard

; 61262b36d0426eaefdb70725-69 (many-objects-room-v1)
; SETUP: 3 dodgeballs, one curved ramp set up against the bin, 3 cylinderblocks and 3 cubeblocks  
; GAMEPLAY: The objective of the game is for the player to score as many points in a row by launching the dodgeballs through the curved ramp into the bin
; SCORING: After each succesful throw, the player sets up a cylinderblock as a way to keep track of scoring. After 4 succesful throws the player earns a cubeblock.
; DIFFICULTY: 2

(define (game 61262b36d0426eaefdb70725-69) (:domain many-objects-room-v1)  ; 69
(:setup (and 
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin) (game-conserved (adjacent ?c ?h)))
))
(:constraints (and 
    (preference ballThrownThroughRampToBin (exists (?d - dodgeball ?c - curved_wooden_ramp ?h - hexagonal_bin)
        (then 
            (once (agent_holds ?d))
            (hold-while 
                (and (not (agent_holds ?d)) (in_motion ?d))
                (touch ?d ?c)    
            )
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:scoring maximize
    (count-nonoverlapping ballThrownThroughRampToBin)
))

; 5fbbf3f438be4c025df6cdd4-70 (many-objects-room-v1)
; SETUP: Take the bin and put it on the floor, in front of the computer, just in the same place where the chair is (put away the chair first). Then take these objects to throw: 3 dodgeballs, 3 golfballs, 3 triangle blocks and 3 pyramid blocks and put it next to the drawer where the alarm clock is, near the bed. Now take the Curved Ramp and put it in front of the Bin, creating a route for the golfballs for more fun.
; GAMEPLAY: With the setup ready, place yourself just in front of the bed, you can put a pillow on the floor as a reference. The objects to throw should be next to you, on your left. To score points throw the objects inside the bin without getting closer to it. You can stand up or you can crouch. If you hit the computer desktop or the laptop there is a penalty.
; SCORING: Triangle blocks give 1 point if they fall inside the bin, pyramid blocks give 2 points, dodgeballs give 2 points, golfballs 3 points. If you use the curved ramp when throwing a golfball (only golfballs) you get 6 points. If you hit the computer or laptop you lose 1 point and you can go below 0 points. So you can end the game with -2 for example.
; DIFFICULTY: 3
(define (game 5fbbf3f438be4c025df6cdd4-70) (:domain many-objects-room-v1)  ; 70
(:setup (and 
    (forall (?c - chair) (game-conserved (not (adjacent_side desk front ?c))))
    (exists (?h - hexagonal_bin ?c - curved_wooden_ramp ) 
        (game-conserved (and 
            (adjacent_side desk front ?c)
            (adjacent_side ?h front ?c back)
        ))
    )
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block))
        (game-optional (< (distance side_table ?o) 1))
    )
))
(:constraints (and 
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block)) (and 
        (preference objectLandsInBin (exists (?h - hexagonal_bin)
            (then 
                (once (and (adjacent agent bed) (agent_holds ?o)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )
        ))
        (preference thrownObjectHitsComputer (exists (?c - (either desktop laptop))
            (then 
                (once (and (adjacent agent bed) (agent_holds ?o)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (touch ?o ?c))
            )
        ))
    ))
    (preference golfballLandsInBinThroughRamp (exists (?g - golfball ?c - curved_wooden_ramp ?h - hexagonal_bin)
        (then 
            (once (and (adjacent agent bed) (agent_holds ?g)))
            (hold-while 
                (and (in_motion ?g) (not (agent_holds ?g)))
                (touch ?c ?g)    
            )
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
(:scoring maximize (+ 
    (count-nonoverlapping objectLandsInBin:triangle_block)
    ( * 2 (count-nonoverlapping objectLandsInBin:pyramid_block))
    ( * 2 (count-nonoverlapping objectLandsInBin:dodgeball))
    ( * 3 (count-nonoverlapping objectLandsInBin:golfball))
    ( * 3 (count-nonoverlapping golfballLandsInBinThroughRamp))
    (* (- 1) (count-nonoverlapping thrownObjectHitsComputer))
)))

; 60a696c3afad1b7f16b0c744-71 (many-objects-room-v1)
; SETUP: It's necessary for the player to position the pillows on the bad as well as the bridges on the floor. It's also necessary for the cylinder blocks to be positioned in challenging spots.
; GAMEPLAY: The game consists of a player, throwing dodgeball to hit the pillows on the bed or using the golfballs to get the ball to go under the little colored bridges. They must throw from the wooden triangle ramp close to the desk and can't hit the colored cylinder blocks, if they do, that throw doesn't count. The player has 10 turn to do the max score he can (the pillow and bridges both score 1 point) 
; SCORING: If the player hits the standing up pillows on the bed with the dodgeball, or throws a golf ball to go under the colored bridge blocks they score 1 point. If the dodgeball or golf ball hits 1 or more cylinders, even after hitting the objective, the throw doesn't count.
; DIFFICULTY: 2
(define (game 60a696c3afad1b7f16b0c744-71) (:domain many-objects-room-v1)  ; 71
(:setup (and 
    (forall (?p - pillow) (game-conserved (on bed ?p)))
    (forall (?b - bridge_block) (game-conserved (on floor ?b)))
    (forall (?c - cylindrical_block) (game-conserved (exists (?o - (either pillow bridge_block)) (< (distance ?c ?o) 1))) )
))
(:constraints (and 
    (preference dodgeballHitsPillowWithoutTouchingBlock (exists (?d - dodgeball ?p - pillow ?r - triangular_ramp) 
        (then 
            (once (and (adjacent agent ?r) (< (distance ?r desk) 1) (agent_holds ?d)))
            (hold-while 
                (and (in_motion ?d) (not (agent_holds ?d)) (not (exists (?c - cylindrical_block) (touch ?c ?d) )) )
                (touch ?d ?p)    
            )
            (once (not (in_motion ?d)))
        )
    ))
    (preference golfballUnderBridgeWithoutTouchingBlock (exists (?g - golfball ?b - bridge_block ?r - triangular_ramp) 
        (then 
            (once (and (adjacent agent ?r) (< (distance ?r desk) 1) (agent_holds ?g)))
            (hold-while 
                (and (in_motion ?g) (not (agent_holds ?g)) (not (exists (?c - cylindrical_block) (touch ?c ?g) )) )
                (above ?g ?b)    
            )
            (once (not (in_motion ?g)))
        )
    ))
))
(:scoring maximize (+ 
    (count-nonoverlapping dodgeballHitsPillowWithoutTouchingBlock)
    (count-nonoverlapping golfballUnderBridgeWithoutTouchingBlock)
)))

; 5fa23c9b64b18a4067cc842e-72 (many-objects-room-v1)
; SETUP: this game is throwing balls at the teddy bear.
; initial setup requires:
; 1) put the teddy bear in an upright position on the bed.
; 2) the 7 balls (3 dodge balls, 3 golf balls, and 1 beach ball) are put together in one place below the chair at the computer desk.
; 3) the player sit on the chair.
; GAMEPLAY: to play the game, the player throws the balls one by one at the teddy bear to make it fall.
; SCORING: each time if the player hits the teddy bear and make is fall (not in an upright position) the player gains 1 point.
; The player must score at least 7 balls to win and lose otherwise.
; DIFFICULTY: 2
(define (game 5fa23c9b64b18a4067cc842e-72) (:domain many-objects-room-v1)  ; 72
(:setup (and 
    (exists (?t - teddy_bear) (game-optional (and (on bed ?t) (object_orientation ?t upright))))
    (forall (?b - ball) (game-optional (< (distance ?b desk) 1)))
))
(:constraints (and  
    (preference ballKnocksTeddy (exists (?b - ball ?t - teddy_bear ?c - chair)
        (then 
            (once (and 
                (on ?c agent)
                (adjacent ?c desk)
                (agent_holds ?b)
                (object_orientation ?t upright)
            ))
            (hold-while 
                (and (in_motion ?b) (not (agent_holds ?b)))
                (touch ?b ?t)    
            )
            (once (not (object_orientation ?t upright)))
        )
    ))
))
(:terminal
    (>= (count-nonoverlapping ballKnocksTeddy) 7)
)
(:scoring maximize
    (count-nonoverlapping ballKnocksTeddy)
))

; 60ef5b1cf52939a80af77543-73 (many-objects-room-v1)
; SETUP: At first you have to place the bin to the centre of the room, and frezze it, then prepare all the dogdeballs near to you.
; GAMEPLAY: You have to stand as close as you can to the table, and try to throw the dogdeballs and the golfballs in to the bin. 
; SCORING: If you succesfully throw a dodgeball to the bin you get a score.
; DIFFICULTY: 3
(define (game 60ef5b1cf52939a80af77543-73) (:domain many-objects-room-v1)  ; 73
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (< (distance ?h room_center) 1)))
    (forall (?d - dodgeball) (game-optional (on desk ?d)))
))
(:constraints (and 
    (preference dodgeballThrownToBinFromDesk (exists (?d - dodgeball ?h - hexagonal_bin) 
        (then 
            (once (and (adjacent agent desk) (agent_holds ?d)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:scoring maximize
    (count-nonoverlapping dodgeballThrownToBinFromDesk)
))

; 613bd3a683a2ac56a4119aa6-74 (many-objects-room-v1)
; SETUP: You must play with the golf balls, the bin must be frozen, also one yellow pillow must be placed in the middle of the room, frozen too
; GAMEPLAY: Throwing golf balls on the bin, trying to make them fall inside of it from a short distance, set by a yellow pillow. It must be placed oin the middle of the room, right infront of the bin. You have to throw 10 times a golf ball from the yellow pillow. If they fall on the bin you score.
; SCORING: There would be 10 chances of throwing. Each ball that falls on the bin would give you 5 points
; DIFFICULTY: 3
(define (game 613bd3a683a2ac56a4119aa6-74) (:domain many-objects-room-v1)  ; 74
(:setup (and 
    (game-conserved (exists (?h - hexagonal_bin ?p - pillow) (< (distance ?h ?p) 3)))
))
(:constraints (and 
    (preference golfballInBinFromPillow (exists (?g - golfball ?h - hexagonal_bin ?p - pillow) 
        (then 
            (once (and (adjacent agent ?p) (agent_holds ?g) (is_setup_object ?p) ))
            (hold (and (in_motion ?g) (not (agent_holds ?g))))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
    (preference throwAttempt (exists (?g - golfball)
        (then
            (once (agent_holds ?g))
            (hold (and (in_motion ?g) (not (agent_holds ?g))))
            (once (not (in_motion ?g)))
        )
    ))
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 10)
)
(:scoring maximize
    (* 5 (count-nonoverlapping golfballInBinFromPillow))
))

; 612fc78547802a3f177e0d53-75 (few-objects-room-v1)
; SETUP: The setup in the room is as it appeared. No additional requirements to play the game.
; GAMEPLAY: Picking up and dropping off the round ball into the bin basket. The player has five attempts to get a 5 star rating. If, after five attempts, the player is not able to successfully drop the ball in the basket, the player loses the game.
; SCORING: The player gets 5 star when the ball is successfully dropped in the bin basket. The player has five attempts to achieve the 5 star rating. 
; DIFFICULTY: 3
(define (game 612fc78547802a3f177e0d53-75) (:domain few-objects-room-v1)  ; 75
(:constraints (and 
    (preference ballDroppedInBin (exists (?b - ball ?h - hexagonal_bin) 
        (then 
            (once (and (adjacent agent ?h) (agent_holds ?b)))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    (preference dropAttempt (exists (?b - ball ?h - hexagonal_bin) 
        (then 
            (once (and (adjacent agent ?h) (agent_holds ?b)))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (not (in_motion ?b)))
        )
    ))
))
(:terminal (or 
    (>= (count-nonoverlapping dropAttempt) 5)
    (>= (count-nonoverlapping ballDroppedInBin) 1)
))
(:scoring maximize
    (* 5 (count-nonoverlapping ballDroppedInBin))
))

; 5d0ba121619661001a7f4fe6-76 (few-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: To play may game, you throw the 6 CubeBlocks in the bin. You stand on the carpet, either on the yellow spot or the pink spot.
; The CubeBlocks are quite big so you end up with a pile of blocks: some of them are inside the bin and some of them are over it resting on top of the ones that are inside.
; You have at most 18 tries to throw all the CubeBlocks succesfully (a throw is succesful when the block ends inside the bin or on the pile) 
; When you have thrown all the CubeBlocks, you stand on the yellow spot and throw the two Dodgeballs at the pile, trying to knock over to the floor as many CubeBlocks as possible.
; SCORING: For each succesful throw of a CubeBlock you get 10 points if you threw the block from the pink spot, or 15 points if you threw the block from the pink spot. 
; If you threw all the blocks succesfully from the yellow spot you get an additional of 15 points.
; If you manage to throw all of the 6 CubeBlocks succesfully in 18 tries or less you get an additional of 15 points.
; Finally, for each block you knock over using the two Dodgeballs you get 20 points.
; DIFFICULTY: 3
(define (game 5d0ba121619661001a7f4fe6-76) (:domain few-objects-room-v1)  ; 76
(:constraints (and 
    (forall (?c - (either pink yellow)) (and 
        (preference blockToBinFromRug (exists (?b - cube_block ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?b) (rug_color_under agent ?c)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and 
                    (not (in_motion ?b)) 
                    (or 
                        (in ?h ?b)
                        (exists (?bl - building) (and 
                            (in ?bl ?b)
                            (in ?h ?bl)
                        ))  
                    )
                ))
            )
        ))
        (preference blockThrowAttempt (exists (?b - cube_block)
            (then 
                (once (and (agent_holds ?b) (rug_color_under agent ?c)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        ))
    ))
    (preference blockKnockedFromBuildingInBin (exists (?d - dodgeball ?h - hexagonal_bin ?bl - building ?b - block)
        (then
            (once (and 
                (agent_holds ?d)
                (rug_color_under agent yellow)
                (in ?bl ?b)
                (in ?h ?bl)
            ))
            (hold-while  
                (and (in_motion ?d) (not (agent_holds ?d)))
                (touch ?d ?b)
                (in_motion ?b)    
            )
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (not (in ?bl ?b))))
        )
    ))
    (preference ballThrowAttempt (exists (?d - dodgeball)
        (then 
            (once (and (agent_holds ?d) (rug_color_under agent yellow)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (not (in_motion ?d)))
        )
    ))
))
(:terminal (and 
    (or 
        (>= (count-once-per-objects blockToBinFromRug) 6)
        (>= (count-nonoverlapping blockThrowAttempt) 18)
    )
    (>= (count-nonoverlapping ballThrowAttempt) 2)  
))
(:scoring maximize (+ 
    (* 10 (count-once-per-objects blockToBinFromRug:pink))
    (* 10 (count-once-per-objects blockToBinFromRug:yellow))
    (* 15 (= (count-once-per-objects blockToBinFromRug:yellow) 6))
    (* 15 (<= (count-nonoverlapping blockThrowAttempt) 18))
    (* 20 (count-once-per-objects blockKnockedFromBuildingInBin))
)))

; 616da508e4014f74f43c8433-77 (many-objects-room-v1)
; SETUP: To prepare the room for the game you have to pick the bin and move it to a place you want it, it can be moved next to you or a few feet away from you, also, you have to pick and carry the dodgeballs with you from the distance you will throw them in purpose to insert them into the bin, the farder you insert the dodgeballs the more points you will gain.
; GAMEPLAY: You will play a game where you will insert the dodgeballs into the bin from the distance you consider you will insert all the dodgeballs into the bin, but, the farder you insert the dodgeballs the more points you will gain.  (The minimun close distance to the bin is one step far from it)
; SCORING: First you will find a position for the bin, so, after that, every step you are far from the bin it will represent the points you will gain if you insert one of the dodgeballs into the bin (after you walk a step for a single direction the next step you walk it has to be different to the opposite step you did before), so, when you are ready  you will try to insert the three dodgeballs balls into the bin. Example, I already positioned the bin where i wanted and I walked three steps to the right and three steps to the front far from the bin, so by then if I choose to throw the dodgeballs from here, each of the balls that I insert into the bin will gain me 6 points.  (The minimun close distance to the bin is one step far from it)
; DIFFICULTY: 2
(define (game 616da508e4014f74f43c8433-77) (:domain many-objects-room-v1)  ; 77
(:constraints (and 
    (preference throwToBinFromDistance (exists (?d - dodgeball ?h - hexagonal_bin)
        (then 
            (once-measure (agent_holds ?d) (distance agent ?h))
            (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
)) 
(:scoring maximize (count-nonoverlapping-measure throwToWallAndBack)
))

; 5eeb326764eb142830aa5cfb-78 (medium-objects-room-v1)
; SETUP: The Teddy bear needs to be moved to the floor  and placed at  the left foot of the bed. It has to be in a sitting position and facing the dog bed. The beachball needs to be placed on the floor near the wall at the right foot of the bed . The bin and basketball need to be moved from that area and placed near the drawer 
; GAMEPLAY: The player stands near the dog bed and throws the dodgeball at the beachball which is behind the teddy bear. The objective of the game is to move the beach ball from its position without disturbing the teddy bear from its sitting position.
; SCORING: The player scores 3 points for every throw in the beachball moves without the teddy being displaced from its sitting position .The player scores -1 if the teddy bear is displaced from its sitting position even if the beachball moves
; DIFFICULTY: 2
(define (game 5eeb326764eb142830aa5cfb-78) (:domain medium-objects-room-v1)  ; 78
(:setup (and 
    (exists (?t - teddy_bear) (game-optional (and 
        (adjacent_side bed front_left_corner ?t)
        (object_orientation ?t upright)   
    )))
    (exists (?b - beachball) (game-optional (and 
        (< (distance_side bed front_left_corner ?b) 1)
        (on floor ?b)
    )))
    (forall (?o - (either hexagonal_bin basketball)) 
        (game-conserved (< (distance ?o side_table) 1))
    )
))
(:constraints (and 
    (preference throwMovesBeachballWithoutKnockingTeddy (exists (?d - dodgeball ?b - beachball ?t - teddy_bear ?db - doggie_bed)
        (then 
            (once (and (agent_holds ?d) (< (distance agent ?db) 1) (object_orientation ?t upright)))
            (hold-while 
                (and (in_motion ?d) (not (agent_holds ?d)) (not (agent_holds ?t)))
                (touch ?d ?b)
                (in_motion ?b)    
            )
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (object_orientation ?t upright)))
        )
    ))
    (preference throwKnocksOverBear (exists (?d - dodgeball ?b - beachball ?t - teddy_bear ?db - doggie_bed)
        (then 
            (once (and (agent_holds ?d) (< (distance agent ?db) 1) (object_orientation ?t upright)))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (agent_holds ?t))))
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (not (object_orientation ?t upright))))
        )
    ))
))
(:scoring maximize (+ 
    (* 3 (count-nonoverlapping throwMovesBeachballWithoutKnockingTeddy))
    (* (- 1) (count-nonoverlapping throwKnocksOverBear))
)))

; 5ba855d47c0ebe0001272f70-79 (many-objects-room-v1)
; SETUP: The setup for the game is fairly simple. There are only two objects needed to play the game: the bin and the golf balls.
; GAMEPLAY: The player will be provided with golf balls and a bin, situated at a far spot of the room. The balls will have to be thrown from a pre-determined distance. The goal is to throw the balls inside the bin.
; SCORING: A player scores a point every time a throw ends inside the bin. Since the number of golf balls is limited, the player will have to pick up the balls after throwing them in order to continue playing. 
; DIFFICULTY: 0
(define (game 5ba855d47c0ebe0001272f70-79) (:domain many-objects-room-v1)  ; 79
(:constraints (and 
    (preference throwGolfballToBin (exists (?g - golfball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?g))
            (hold (and (not (agent_holds ?g)) (in_motion ?g)))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
(:scoring maximize (count-nonoverlapping throwGolfballToBin)
))

; 5ea3a20ac30a773368592f9e-80 (few-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: To play my game you should move objects to the center of the room based on a color order. The color order is Pink, Blue, Brown, Multi-Color (Plaid), Green, then Tan.
; SCORING: You will receive one point for every object moved in the correct order. 
; DIFFICULTY: 1
(define (game 5ea3a20ac30a773368592f9e-80) (:domain few-objects-room-v1)  ; 80
(:constraints (and 
    (preference pinkObjectMovedToRoomCenter (exists (?o - game_object)
        (then 
            (once (and (agent_holds ?o) (= (color ?o) pink)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)))
        )
    ))
    (preference blueObjectMovedToRoomCenter (exists (?o - game_object)
        (then 
            (once (and (agent_holds ?o) (= (color ?o) blue)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 - game_object) (and 
                    (= (color ?o1) pink) (< (distance room_center ?o1) 1)  
                ))
            ))
        )
    ))
    (preference brownObjectMovedToRoomCenter (exists (?o - game_object)
        (then 
            (once (and (agent_holds ?o) (= (color ?o) brown)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 - game_object) (and 
                    (= (color ?o1) pink) (< (distance room_center ?o1) 1)  
                    (= (color ?o2) blue) (< (distance room_center ?o2) 1)  
                ))
            ))
        )
    ))
    (preference pillowMovedToRoomCenter (exists (?o - pillow)
        (then 
            (once (and (agent_holds ?o)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 ?o3 - game_object) (and 
                    (= (color ?o1) pink) (< (distance room_center ?o1) 1)  
                    (= (color ?o2) blue) (< (distance room_center ?o2) 1)  
                    (= (color ?o3) brown) (< (distance room_center ?o3) 1)  
                ))
            ))
        )
    ))
    (preference greenObjectMovedToRoomCenter (exists (?o - game_object)
        (then 
            (once (and (agent_holds ?o) (= (color ?o) green)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 ?o3 ?o4 - game_object) (and 
                    (= (color ?o1) pink) (< (distance room_center ?o1) 1)  
                    (= (color ?o2) blue) (< (distance room_center ?o2) 1)  
                    (= (color ?o3) brown) (< (distance room_center ?o3) 1)  
                    (= (type ?o4) pillow) (< (distance room_center ?o4) 1)  
                ))
            ))
        )
    ))
    (preference tanObjectMovedToRoomCenter (exists (?o - game_object)
        (then 
            (once (and (agent_holds ?o) (= (color ?o) tan)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 ?o3 ?o4 ?o5 - game_object) (and 
                    (= (color ?o1) pink) (< (distance room_center ?o1) 1)  
                    (= (color ?o2) blue) (< (distance room_center ?o2) 1)  
                    (= (color ?o3) brown) (< (distance room_center ?o3) 1)  
                    (= (type ?o4) pillow) (< (distance room_center ?o4) 1)  
                    (= (color ?o5) green) (< (distance room_center ?o5) 1)  
                ))
            ))
        )
    ))
))
(:scoring maximize (+ 
    (count-once pinkObjectMovedToRoomCenter)
    (count-once blueObjectMovedToRoomCenter)
    (count-once brownObjectMovedToRoomCenter)
    (count-once pillowMovedToRoomCenter)
    (count-once greenObjectMovedToRoomCenter)
    (count-once tanObjectMovedToRoomCenter)
)))

; 5fdee4d96a36576ca62e4518-81 (many-objects-room-v1)
; SETUP: Put the bin any where by the desk by the two posters. Put the ramps or other objects around the bin but keep the bin free from any objects directly in front of it.
; GAMEPLAY: Take the three dodgeballs and try to get it in the bin. You must stand on the carpet. Anything over the carpet is out of bounds. Try different angles and power to create cool shots. 
; SCORING: Each ball in the bin counts as one point. 3 is the goal.  
; DIFFICULTY: 3
(define (game 5fdee4d96a36576ca62e4518-81) (:domain many-objects-room-v1)  ; 81
(:setup (and 
    (exists (?h - hexagonal_bin ?r1 ?r2 - (either triangular_ramp curved_wooden_ramp)) 
        (game-conserved (and 
            (adjacent ?h desk)
            (< (distance ?h ?r1) 1)
            (< (distance ?h ?r2) 1)
            (not (exists (?o - game_object) (adjacent_side ?h front ?o)))
        ))
    )
))
(:constraints (and 
    (preference dodgeballFromRugToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then 
            (once (and (agent_holds ?d) (on rug agent)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:terminal 
    (>= (count-nonoverlapping dodgeballFromRugToBin) 3)
)
(:scoring maximize
    (count-nonoverlapping dodgeballFromRugToBin)
))

; 6172378d423fdf1acdc2d212-82 (many-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: I would like to play how many balls can you drop/throw in the bin within 5 minutes. You can stand anywhere in the room to get the ball into the bin.
; SCORING: Each ball that one throws in the bin is 1 point.
; DIFFICULTY: 4
(define (game 6172378d423fdf1acdc2d212-82) (:domain many-objects-room-v1)  ; 82
(:constraints (and 
    (preference ballThrownToBin (exists (?b - ball ?h - hexagonal_bin)
        (then 
            (once (agent_holds ?b))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
))
(:terminal 
    (>= (total-time) 300)
)
(:scoring maximize
    (count-nonoverlapping ballThrownToBin)
))

; 5bdfb648484288000130dad0-83 (many-objects-room-v1)
; SETUP: Put the bin in between the two computer chairs laying with opening horizontal to the ground grab a dodgeball and go to the edge of the bed
; GAMEPLAY: standing at the edge of the bed attempt to throw or bounce the ball into the bin
; SCORING: each dodgeball is worth 1 point you get three chances to score 3 points if you make all the dodge balls then you use golf balls to add to that score each worth 1 point so in a perfect game you could score 6 points
; DIFFICULTY: 3
(define (game 5bdfb648484288000130dad0-83) (:domain many-objects-room-v1)  ; 83
(:setup (and 
    (exists (?h - hexagonal_bin ?c1 ?c2 - chair) (game-conserved (and 
        (object_orientation ?h sideways)
        (between ?c1 ?h ?c2)
    )))
))
(:constraints (and
    (forall (?b - (either dodgeball golfball))
        (preference ballToBinFromBed (exists (?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?b) (adjacent bed agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
(:scoring maximize (+ 
    (count-once-per-objects ballToBinFromBed:dodgeball)
    (* (= (count-once-per-objects ballToBinFromBed:dodgeball) 3) (count-once-per-objects ballToBinFromBed:golfball))
)))

; 84 is a hiding game -- invalid

; 61272733b6c8fe076880e02c-85 (few-objects-room-v1)
; SETUP: I think there's an obvious and easy to setup throwing game with the objects in this room. Presented here you can use the trash can and the cubes/balls, but any sort of safe, toss able object and receptacle should work.
; GAMEPLAY: Stand in the pink square on the carpet and pick up one of the six cubes. Aim for the bin and throw. each cube. You get one chance with each cube, and if you do make it into the bin, you need to remove the cube.
; SCORING: The light beige cubes are worth 1 point. The dark beige cubes are worth 2, and the blue cubes are worth 3. Your total score is the total at the end of throwing the cubes. You lose one point when you throw a cube, but you cannot go into the negatives. You can pick the cubes in any order.
; DIFFICULTY: 1
(define (game 61272733b6c8fe076880e02c-85) (:domain few-objects-room-v1)  ; 85
(:constraints (and 
    (forall (?c - color)
        (preference cubeThrownToBin (exists (?h - hexagonal_bin ?b - cube_block)
            (then 
                (once (and 
                    (agent_holds ?b) 
                    (rug_color_under agent pink) 
                    (= (color ?b) ?c)
                    (not (exists (?ob - cube_block) (in ?h ?ob)))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
    (forall (?b - cube_block)
        (preference throwAttempt 
            (then 
                (once (and 
                    (agent_holds ?b) 
                    (rug_color_under agent pink) 
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal (or 
    (>= (count-maximal-nonoverlapping throwAttempt) 2)
    (>= (count-once-per-objects throwAttempt) 6)
))
(:scoring maximize (+ 
    (count-once-per-objects cubeThrownToBin:yellow)
    (* 2 (count-once-per-objects cubeThrownToBin:tan))
    (* 3 (count-once-per-objects cubeThrownToBin:blue))
    (* (- 1) (count-once-per-objects throwAttempt))
)))

; 86 is a dup of 84 -- and is aldo invalid

; 6158d01f302cf46b673dd597-87 (few-objects-room-v1)
; SETUP: Place the waste basket on the desk against the wall
; GAMEPLAY: you take 1 of 2 objects - either a dodgeball or a play block. you must stand on the carpet and throw either one of these objects towards the waste basket on the table in an attempt to get either object (dodgeball or play block) inside the trash can opening
; SCORING: 2 points for successfully scoring a "basket" with a dodgeball, 1 point for successfully scoring a "basket" with a block
; DIFFICULTY: 4
(define (game 6158d01f302cf46b673dd597-87) (:domain few-objects-room-v1)  ; 87
(:setup (and 
    (exists (?h - hexagonal_bin ?w - wall) (game-conserved (and 
        (on desk ?h)
        (adjacent ?h ?w)
    )))
))
(:constraints (and 
    (forall (?o - (either dodgeball block)) 
        (preference basketMadeFromRug (exists (?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?o) (on rug agent)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )
        ))
    )
))
(:scoring maximize (+ 
    (count-nonoverlapping basketMadeFromRug:dodgeball)
    (* 2 (count-nonoverlapping basketMadeFromRug:block))
)))

; 5fefd5b2173bfbe890bc98ed-88 (few-objects-room-v1)
; SETUP: Place the bucket on the bed and round pillow, slightly angled  so it is tilted to face the computer at a 45 degree angle. Take the 6 blocks from under the shelf and stack 3 on both sides of the bucket. take a ball from the shelf to use as the game ball. I prefer the pink ball. The throwing line is the back edge of the carpet. 
; GAMEPLAY: Try to make 5 baskets out of 10 throws. 
; SCORING: One basket equals one point. You only have 10 throws in total, and three misses in a row ends the game as a loss. Alternatively, missing the bucket and knocking over either of the cube stacks on either side will wipe your score clean and end the game as a loss. To win the game, you must make 5 baskets out of 10 tosses. 
; DIFFICULTY: 3
(define (game 5fefd5b2173bfbe890bc98ed-88) (:domain few-objects-room-v1)  ; 88
(:setup (and 
    (exists (?h - hexagonal_bin ?p - pillow ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
        (game-conserved (and 
            (on bed ?h)
            (not (object_orientation ?p sideways))
            (not (object_orientation ?p upright))
            (not (object_orientation ?p upside_down))
            (adjacent_side ?h left ?b1)
            (on bed ?b1)
            (on ?b1 ?b2)
            (on ?b2 ?b3)
            (adjacent_side ?h right ?b4)
            (on bed ?b4)
            (on ?b4 ?b5)
            (on ?b5 ?b6)
        ))
    )
))
(:constraints (and 
    (preference throwFromEdgeOfRug (exists (?d - dodgeball ?h - hexagonal_bin)
        (then 
            (once (and 
                (agent_holds ?d) 
                (on floor agent)
                (adjacent rug agent)
                (> (distance agent bed) 2)
            ))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
    (preference throwAttempt (exists (?d - dodgeball)
        (then 
            (once (and 
                (agent_holds ?d) 
                (on floor agent)
                (adjacent rug agent)
                (> (distance agent bed) 2)
            ))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (not (in_motion ?d)))
        )
    ))
    (preference throwAttemptKnocksBlock (exists (?d - dodgeball ?c - cube_block)
        (then 
            (once (and 
                (agent_holds ?d) 
                (on floor agent)
                (adjacent rug agent)
                (> (distance agent bed) 2)
            ))
            (hold-while 
                (and (in_motion ?d) (not (agent_holds ?d)))
                (touch ?d ?c)
                (in_motion ?c)
            )
            (once (not (in_motion ?d)))
        )
    ))
))
(:terminal (or 
    (>= (count-nonoverlapping throwAttempt) 10)
    ; TODO: there's also a "streak of three misses ends the game" constraint that I'm currently omittting
    (>= (count-once throwAttemptKnocksBlock) 1)
    (>= (total-score) 5)
))
(:scoring maximize 
    (count-nonoverlapping throwFromEdgeOfRug)
))

; 6103ec2bf88328284fd894bc-89 (medium-objects-room-v1)
; SETUP: Player has to take the monitor off the table using other objects. Player takes the green Bin and places it on the center of the table facing bed side of the room. Bin must be frozen.
; GAMEPLAY: Player has to pick up one of the three balls that are present in the room and throw them using hold mouse button into the Bin. Player has to stand on the rug during throw for points to be counted. There is three minute limit to play the game. The minimum score to win is ten points.
; SCORING: Scoring Dodgeball grants +1 point, scoring basketball grants +2 points. Throwing beachball from rug position to the bin so that ball is standing stable on the bin grants instant win.
; DIFFICULTY: 4
(define (game 6103ec2bf88328284fd894bc-89) (:domain medium-objects-room-v1)  ; 89
(:setup (and 
    (exists (?d - desktop ?h - hexagonal_bin) (game-conserved (and 
        (on desk ?h)
        (not (on desk ?d))
    )))
))
(:constraints (and 
    (forall (?b - ball)
        (preference ballThrownFromRug (exists (?h - hexagonal_bin) 
            (then
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
(:terminal (or 
    (>= (total-time) 180)
    (>= (total-score) 10)
))
(:scoring maximize (+ 
    (count-nonoverlapping ballThrownFromRug:dodgeball)
    (* 2 (count-nonoverlapping ballThrownFromRug:basketball))
    (* 10 (count-nonoverlapping ballThrownFromRug:beachball))
)))

(define (game 5f511e9381da7d30c91a46a2-90) (:domain many-objects-room-v1)  ; 90
(:constraints (and 
    (preference dodgeballBouncesOnceToDoggieBed (exists (?d - dodgeball ?b - doggie_bed)
        (then
            (once (agent_holds ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (touch floor ?d))))
            (once (touch floor ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (touch floor ?d))))
            (once (and (not (in_motion ?d)) (on ?b ?d)))
        )
    ))
))
(:scoring maximize
    (count-nonoverlapping dodgeballBouncesOnceToDoggieBed)
))

; 91 is a dup of 89 with slightly different scoring numbers

; 92 is a hiding game -- invalid

; 60a6ba026f8bd75b67b23c97-93 (many-objects-room-v1)
; SETUP: For this game I would need to have the trash can in some specific spot, but that spot can change. The player would have a dogeball in the hands and try to score it the trash can.
; GAMEPLAY: A game I could play in this room is a classic basket. Just pick up a dogeball and try to hit it inside the trash can.
; SCORING: Each throw inside the trash can would give the player 1 point. Otherwise 0 points.
; DIFFICULTY: 3
(define (game 60a6ba026f8bd75b67b23c97-93) (:domain many-objects-room-v1)  ; 93
(:constraints (and 
    (preference throwBallToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then 
            (once (agent_holds ?d))
            (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:scoring maximize
    (count-nonoverlapping throwBallToBin)
))

; 5cdad620eae6f70019d4e950-94 (many-objects-room-v1)
; SETUP: Leave all objects where they are to start the game.
; GAMEPLAY: The object of the game is to throw the golfballs and the dodgeballs into the trashbin. You pick up a golfball or dodgeball and then stand with your back against the mirror on the door and try to toss it into the bin. You have 8 attempts to make it into the bin. Highest score wins
; SCORING: You get 3 points for each golfball you make into the bin and 6 points for each dodgeball you toss into the bin. 
; DIFFICULTY: 3
(define (game 5cdad620eae6f70019d4e950-94) (:domain many-objects-room-v1)  ; 94
(:constraints (and 
    (forall (?b - (either dodgeball golfball)) (and 
        (preference ballThrownFromDoor (exists (?h - hexagonal_bin) 
            (then
                (once (and (agent_holds ?b) (adjacent door agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        (preference throwAttemptFromDoor 
            (then
                (once (and (agent_holds ?b) (adjacent door agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        )
    ))
))
(:terminal
    (>= (count-nonoverlapping throwAttemptFromDoor) 8)
)
(:scoring maximize (+ 
    (* 3 (count-nonoverlapping ballThrownFromDoor:dodgeball))
    (* 6 (count-nonoverlapping ballThrownFromDoor:golfball))
)))

; 95 requires counting something that happens during a preference

; 96 requires is underconstrainted -- I'm omitting it for now

; 5b6a87d2cda8590001db8e07-97 (medium-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: Player takes the red Dodgeball. They stand anywhere outside the rug in front of the SmallRamp. They throw the Dodgeball so it hits the colors on the rug in front of the SmallRamp. Player has to score as many points as possible in 1 min.
; SCORING: If the Dodgeball hits the colors in front of the SmallRamp, then player gets 1 point.
; DIFFICULTY: 1
(define (game 5b6a87d2cda8590001db8e07097) (:domain medium-objects-room-v1)  ; 97
(:constraints (and 
    (preference ballThrownToRug (exists (?d - red_dodgeball)
        (then
            (once (and (agent_holds ?d) (not (on rug agent))))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (on rug ?d)))
        )
    ))
))
(:terminal 
    (>= (total-time) 60)
)
(:scoring maximize
    (count-nonoverlapping ballThrownToRug)
))

; 5f038dc85819b15b08840dfd-98 (medium-objects-room-v1)
; SETUP: Move the trash bin over so its not under the cabinet, and get the three balls and move them onto your bed.
; GAMEPLAY: You toss the various balls from your bed while laying down and bounce it off of the wall into the trash bin. If the throw the biggest ball in first you won't be able to fit the smaller ones in, so you have to use strategy to get the most points.
; SCORING: If you land a ball into the trash bin you get 1 point for the big ball 2 points for the medium ball and 3 points for the smallest ball. If you get all 3 balls into the trash you get a total of 6 points the maximum and you win.
; DIFFICULTY: 4
(define (game 5f038dc85819b15b08840dfd0-98) (:domain medium-objects-room-v1)  ; 98
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (not (exists (?s - shelf) (above ?h ?s)))))
    (forall (?b - ball) (game-optional (on bed ?b)))
))
(:constraints (and 
    (forall (?b - ball) 
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (or (on bed agent) (adjacent bed agent))))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )  
        ))
    )
))
(:terminal
    (>= (total-score) 6)
)
(:scoring maximize (+ 
    (count-once-per-objects ballThrownToBin:beachball)
    (* 2 (count-once-per-objects ballThrownToBin:basketball))
    (* 3 (count-once-per-objects ballThrownToBin:dodgeball))
)))

; 5fbd9bcc54453f1b0b28d89a-99 (few-objects-room-v1)
; SETUP: N/A
; GAMEPLAY: pick a cubeblock and try to throw it on one of the shelves, you have to stay in front of the bed
; SCORING: the score is 6/6, you have to get one cubeblock in in three attempts
; DIFFICULTY: 1
(define (game 5fbd9bcc54453f1b0b28d89a-99) (:domain few-objects-room-v1)  ; 99
(:constraints (and 
    (preference cubeBlockFromBedToShelf (exists (?c - cube_block ?s - shelf)
        (then 
            (once (and (agent_holds ?c) (adjacent bed agent)))
            (hold (and (in_motion ?c) (not (agent_holds ?c))))
            (once (and (not (in_motion ?c)) (on ?s ?c)))
        )
    ))
    (preference cubeBlockThrowAttempt (exists (?c - cube_block)
        (then 
            (once (and (agent_holds ?c) (adjacent bed agent)))
            (hold (and (in_motion ?c) (not (agent_holds ?c))))
            (once (not (in_motion ?c)))
        )
    ))
))
(:terminal 
    (>= (count-nonoverlapping cubeBlockThrowAttempt) 3)
)
(:scoring maximize
    (count-nonoverlapping cubeBlockFromBedToShelf)
))

; 5c7ceda01d2afc0001f4ad1d-100 (medium-objects-room-v1)
; SETUP: Place the bin on top of the bed and the dog bed on the floor in front of the bed, in line with the bin.
; GAMEPLAY: Standing at the desk, try to throw the dodgeball into either the bin or dog bed as many times as possible within two minutes.
; SCORING: Three points for landing the ball in the bin and two for the dog bed.
; DIFFICULTY: 2

(define (game 5c7ceda01d2afc0001f4ad1d-100) (:domain medium-objects-room-v1)  ; 100
(:setup (and 
    (exists (?h - hexagonal_bin ?d - doggie_bed) (game-conserved (and 
        (on floor ?d)
        (on bed ?h)
        ; TODO: is the below nicer than (= (z_position ?t1) (z_position ?T2))
        (equal_z_position ?h ?d)
    )))
))
(:constraints (and 
    (forall (?t - (either hexagonal_bin doggie_bed)) 
        (preference dodgeballFromDeskToTarget (exists (?d - dodgeball)
            (then 
                (once (and (agent_holds ?d) (adjacent desk agent)))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (or (in ?t ?d) (on ?t ?d))))
            )
        ))
    )
))
(:scoring maximize (+ 
    (* 2 (count-nonoverlapping dodgeballFromDeskToTarget:doggie_bed))
    (* 3 (count-nonoverlapping dodgeballFromDeskToTarget:hexagonal_bin))
)))

; 61093eae2bc2e47e6f26c7d7-101 (few-objects-room-v1)
; SETUP: the bin needs to be ontop op the bed,plcae the chairs out of the way since the would not be of use within the game,place the ramp infront of bed facing the desk (so that ball rolls from ramp to desk if it falls on ramp),place 2 blue blocks about 1metre from desk and yellow blocks about 2 metres from desk infront of blue blocks.
; GAMEPLAY: the balls will be used to try and throw them into the bin that is placed on the bed,by standing behind the blue blocks it will be a further distance to throw,by standing behind the yellow blocks it will be closer to the bin and thus easier to throw balls in.if you miss the bin the balls will fal on the ramp and roll back to the starting point (either behind blue or yellow blocks depending on how difficult you want it to be)
; SCORING: If you throw a ball in from behind the blue blocks it will count as 10 points,where throwing a ball in from behind the yellow blocks will count as 5 points.If you throw both balls in from behind the blue blocks it will count as 30 points and if you throw both balls in from behind the yellow blocks it will count as 15 ponts.Once you miss both balls on a attempt your points will reset and the goal is to reach 50 points to move on to different level.
; DIFFICULTY: 3
(define (game 61093eae2bc2e47e6f26c7d7-101) (:domain few-objects-room-v1)  ; 101
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (on bed ?h)))
    (exists (?r - curved_wooden_ramp) (game-conserved (and (adjacent bed ?r) (faces ?r desk))))
    (exists (?c1 ?c2 - blue_cube_block ?c3 ?c4 - yellow_cube_block) (game-conserved (and 
        (= (distance ?c1 desk) 1)  
        (= (distance ?c2 desk) 1)
        (= (distance ?c3 desk) 2)  
        (= (distance ?c4 desk) 2)
        (between desk ?c1 ?c3)
        (between desk ?c2 ?c4)
    )))
))
(:constraints (and 
    (forall (?c - (either blue_cube_block yellow_cube_block)) (and 
        (preference ballThrownFromBehindBlock (exists (?b - ball ?h - hexagonal_bin)
            (then 
                (once (and 
                    (agent_holds ?b) 
                    (is_setup_object ?c)
                    (>= (distance agent ?h) (distance ?c ?h))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        (preference throwAttemptFromBehindBlock (exists (?b - ball ?h - hexagonal_bin)
            (then 
                (once (and 
                    (agent_holds ?b) 
                    (is_setup_object ?c)
                    (>= (distance agent ?h) (distance ?c ?h))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b))))
            )
        ))
    ))
))
(:terminal (or 
    (>= (count-nonoverlapping throwAttemptFromBehindBlock) 2)
    (>= (total-score) 50)
))
(:scoring maximize (+ 
    (* 10 (count-nonoverlapping ballThrownFromBehindBlock:blue_cube_block))
    (* 5 (count-nonoverlapping ballThrownFromBehindBlock:yellow_cube_block))
    (* 30 (= (count-nonoverlapping ballThrownFromBehindBlock:blue_cube_block) 2))
    (* 15 (= (count-nonoverlapping ballThrownFromBehindBlock:yellow_cube_block) 2))
)))


; 102 is almost a copy of 101 and same participant -- omit

; 5b94d723839c0a00010f88d9-103 (few-objects-room-v1)
; SETUP: trash placed on middle of a carpet, two balls and bed that we can place trash on
; GAMEPLAY: Throwing balls in the trash that is laying on the bed, ten rounds
; SCORING: hitting in the middle for two points and hitting hoop for one point
; DIFFICULTY: 2
(define (game 5b94d723839c0a00010f88d9-103) (:domain few-objects-room-v1)  ; 103
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (and
        (on bed ?h) 
        (object_orientation ?h sideways) 
    )))
))
(:constraints (and 
    (preference dodgeballHitsBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold-while 
                (and (in_motion ?d) (not (agent_holds ?d)) (not (in ?h ?d)))
                (touch ?h ?d)
            )
            (once (and (not (in_motion ?d)) (not (in ?h ?d)))) 
        )
    ))
    (preference dodgeballHitsBinBottom (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold-while 
                (and (in_motion ?d) (not (agent_holds ?d)))
                (in ?h ?d)
            )
            (once (and (not (in_motion ?d)))) 
        )
    ))
    (preference throwAttempt (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)))) 
        )
    ))
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 10)
)
(:scoring maximize (+ 
    (count-nonoverlapping dodgeballHitsBin)
    (* 2 (count-nonoverlapping dodgeballHitsBinBottom))
)))

; 6106ac34408681f3b0d07396-104 (few-objects-room-v1)
; SETUP: You have to move the bin so it lines up with the inner side of the second balcony. You use that line to put the bin in the middle of said line. Freeze it there.
; GAMEPLAY: You have to pick u a dodgeball and try to throw it in the bin standing on the outer edge of the rug.
; SCORING: The dodgeball has to get inside the bin but not necessarily stay there. Each hit counts as one point. You have a set amount of time to complete the challenge. 5 minutes. But it can be increased of course.
; DIFFICULTY: 4
(define (game 6106ac34408681f3b0d07396-104) (:domain few-objects-room-v1)  ; 104
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (and
        (equal_x_position ?h east_sliding_door) 
    )))
))
(:constraints (and 
    (preference throwFromEdgeOfRug (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (agent_holds ?d) (adjacent rug agent)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (in ?h ?d))  ; participant specified that couning out is okay
        )
    ))
))
(:terminal
    (>= (total-time) 300)
)
(:scoring maximize
    (count-nonoverlapping throwFromEdgeOfRug)
))  

; 61015f63f9a351d3171a0f98-105 (few-objects-room-v1)
; SETUP: Move both chairs out of the way, they could go on or near the bed. Pick up and move all 6 cubeBlocks and move them to the colourful matt near the bed. Make sure the desktop has nothing underneath the desk. Throw the dodgeballs as hard as possible to create randomly placed obstacles wherever they land.
; GAMEPLAY: The game is called Save the wood. The rules of the game are to use the blue cubeBlock and the white cubeBlock to create a path from the bed (or as close as the game with allow you to get to the bed to create a path of blocks from the bed to underneath the desktop computer. YOU CANNOT THROW THE WOODEN cubeBlocks (they are to be saved). You can only use the blue and white. You throw them, one after another and they must land extremely near each other so they are touching to create your path to the desktop computer where they (the pure wooden blocks) are saved. Hence the name 'Save the Wood'
; SCORING: The scoring system works as such: When you create one path you leave 1 wooden block (one of the two pure wooden blocks) under the desk. You then restart and (again only using the white or blue blocks) you create a path - throwing blue and white block after blue and white block - to create a path from the bed to underneath the desktop. You have completed the game when you have both wooden blocks under the desk.
; You could even incorporate the 2 dodgeballs into the game and throw them at the beginning of the game, to create obstacles which you must play/play around when creating your path from the bed to underneath the desk.
; DIFFICULTY: 1
(define (game 61015f63f9a351d3171a0f98-105) (:domain few-objects-room-v1)  ; 105
(:setup (and 
    (forall (?c - cube_block) (game-optional (on rug ?c)))
    (game-optional (not (exists (?o - game_object) (above ?o desk))))
    (forall (?d - dodgeball) (game-conserved (not (exists (?s - shelf) (on ?s ?d)))))
))
(:constraints (and 
    (preference woodenBlockMovedFromRugToDesk (exists (?b - tan_cube_block)
        (then 
            (once (and 
                (forall (?c - (either blue_cube_block yellow_cube_block)) (on rug ?c))
                (on rug ?b)
            ))
            (hold (forall (?c - (either blue_cube_block yellow_cube_block)) (or
                (on rug ?c) 
                (agent_holds ?c)
                (in_motion ?c)
                (exists (?c2 - (either blue_cube_block yellow_cube_block)) (and 
                    (not (= ?c ?c2))
                    (< (distance ?c ?c2) 0.5)
                    (on floor ?c)
                    (on floor ?c2) 
                ))
            )))
            (hold (forall (?c - (either blue_cube_block yellow_cube_block))
                (< (distance desk ?c) 1)
            ))
            (once (above ?b desk)) 
        )  
    ))
))
(:scoring maximize
    (count-once-per-objects woodenBlockMovedFromRugToDesk)
))

; 5d67b6d92b7448000173d95a-106 (few-objects-room-v1)
; SETUP: You need to place the bin in a place where it can be used as a basket, and it must be frozen in place
; GAMEPLAY: You have to throw the balls inside the bin
; SCORING: The cubeblocks are used as points. You start with 0 points and your goal is to reach 6. If the ball goes in the bin, you gain 1 point. You have 15 attempts per game
; DIFFICULTY: 4
(define (game 5d67b6d92b7448000173d95a-106) (:domain few-objects-room-v1)  ; 106
(:constraints (and 
    (preference throwInBin (exists (?b - ball ?h - hexagonal_bin)
        (then 
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    (preference throwAttempt (exists (?b - ball)
        (then 
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
            (once (not (in_motion ?b)))
        )
    ))
))
(:terminal (or 
    (>= (total-score) 6)
    (>= (count-nonoverlapping throwAttempt) 15)
))
(:scoring maximize
    (count-nonoverlapping throwInBin)
))

; 107 and 109 are by the same participant, and 109 is actually mostly valid

; 5f0af097e7d15b3bf7734642-108 (medium-objects-room-v1)
; SETUP: Clear the bed and the nightstand of any items, then place the two LongCylinderBlocks onto it, at its extremities, and one CylinderBlock onto the nightstand. Put the two PyramidBlocks onto the LongCylinderBlocks. Place the Bin onto the bed, in between the two LongCylinderBlocks. Put the Beachball, the BasketBall and the Dodgeball onto the DogBed (or next to it).
; GAMEPLAY: Over the course of 3 rounds, you will throw the three balls you have in the room to items placed on the bed. The goal is to knock off the PyramidBlocks without toppling the LongCylinderBlocks, and to put one or more balls in the Bin. You can also knock off the CylinderBlock from the nightstand. The Beachball is too big for the Bin, so you have to throw it in a way that makes it rest onto the top of the bin. You will throw the balls from a position next to the DogBed. If a ball ricochets back in a way that would enable you to pick it up, you can throw it again. A round ends when you have no more balls to throw. After each round, you will have to reset the position of the items on the bed, and to put the balls back onto the DogBed.
; SCORING: Each PyramidBlock knocked off is +3 points for the player. Each LongCylinderBlock toppled while trying to knock PyramidBlocks off is -3 points. Knocking the CylinderBlock from the nightstand off is +1 point. Putting the Dodgeball or the Basketball into the bin is +2 point. Throwing the Beachball into the Bin is +4 points. Balls that miss net 0 points; that includes having the Beachball slide off the top of the Bin.
; DIFFICULTY: 3
(define (game 5f0af097e7d15b3bf7734642-108) (:domain medium-objects-room-v1)  ; 108
(:setup (and 
    (exists (?h - hexagonal_bin ?b1 ?b2 - tall_cylindrical_block ?p1 ?p2 - pyramid_block ?b3 - cylindrical_block) 
        (and 
            (game-conserved (and 
                (on side_table ?b3)
                (on bed ?b1)
                (on ?b1 ?p1)
                (on bed ?b2)
                (on ?p2 ?b2)
                (adjacent ?b1 north_wall)
                (between ?b1 ?h ?b2)
                (= (distance ?b1 ?h) (distance ?b2 ?h))
            ))
            (game-optional (and 
                (on bed ?h)
                (equal_z_position bed ?h)
            ))   
        )  
    )
    (exists (?d - doggie_bed) (forall (?b - ball) (game-optional (or 
        (on ?d ?b)
        (< (distance ?d ?b) 0.5)
    ))))
))
(:constraints (and 
    (preference agentLeavesDogbedOrNoMoreBalls (exists (?d - doggie_bed)
        (then
            (hold (<= (distance ?d agent) 1))
            (once (or 
                (> (distance ?d agent) 1)
                (forall (?b - ball) (and 
                    (not (in_motion ?b))
                    (> (distance agent ?b) 1))
                )
            ))
        )
    ))
    (forall (?c - (either cylindrical_block tall_cylindrical_block pyramid_block))
        (preference throwKnocksBlock (exists (?b - ball ?d - doggie_bed)
            (then
                (once (and 
                    (is_setup_object ?c)
                    (agent_holds ?b)
                    (<= (distance ?d agent) 1)
                ))
                (hold-while 
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )    
            )
        ))
    )
    (forall (?b - ball) 
        (preference ballInOrOnBin (exists (?d - doggie_bed ?h - hexagonal_bin)
            (then 
                (once (and 
                    (agent_holds ?b)
                    (<= (distance ?d agent) 1)
                ))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (or (in ?h ?b) (on ?h ?b))))
            )
        ))
    )
))
(:terminal
    (>= (count-once agentLeavesDogbedOrNoMoreBalls) 1)
)
(:scoring maximize (+ 
    (* 3 (count-once-per-external-objects throwKnocksPyramidBlock:pyramid_block))
    (* (- 3) (count-once-per-external-objects throwKnocksPyramidBlock:tall_cylindrical_block))
    (count-once-per-external-objects throwKnocksPyramidBlock:cylindrical_block) 
    (* 2 (count-once-per-external-objects ballInOrOnBin:dodgeball))
    (* 2 (count-once-per-external-objects ballInOrOnBin:basketball))
    (* 4 (count-once-per-external-objects ballInOrOnBin:beachball))
)))

; 5f9aba6600cdf11f1c9b915c-109 (many-objects-room-v1)
; SETUP: You will need: balls, wooden triangles, a garbage can, pillows, a white circle, wooden cubes
; GAMEPLAY: You can play throwing the ball into the trash can, throwing pillows to the designated object (white circle), throwing cubic dice on the top shelf
; SCORING: One point is awarded for every single task completed
; DIFFICULTY: 0
(define (game 5f9aba6600cdf11f1c9b915c-109) (:domain many-objects-room-v1)  ; 109
(:constraints (and 
    (preference ballThrownToBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    (preference cubeBlockThrownToTopShelf (exists (?c - cube_block)
        (then
            (once (agent_holds ?c))
            (hold (and (not (agent_holds ?c)) (in_motion ?c)))
            (once (and (not (in_motion ?c)) (on top_shelf ?c)))
        )
    ))
    (preference pillowThrownToDoggieBed (exists (?p - pillow ?d - doggie_bed)
        (then
            (once (agent_holds ?p))
            (hold (and (not (agent_holds ?p)) (in_motion ?p)))
            (once (and (not (in_motion ?p)) (on ?d ?p)))
        )
    ))
))
(:scoring maximize (+ 
    (count-once-per-objects ballThrownToBin)
    (count-once-per-objects cubeBlockThrownToTopShelf)
    (count-once-per-objects pillowThrownToDoggieBed)
)))

; 6123dcdd95e4f8afd71928a3-110 (few-objects-room-v1)
; SETUP: First you need to cut the room in half. For this purpose pick up the two Chairs and place them in front of the door, one close to the door and one close to the wall on the other side of the room. Than place the Bin to the corner under the two balls facing it's lowest side to the corner. Finally collect 10 items and bring themm in front of the computer desk - the items: 2 Dodgeballs, 6 Cubeblocks, 1 Alarmclock, 1 Book.
; GAMEPLAY: This game is a basic basketball game. Your goal is to toss the 10 items into the Bin. You can toss only from behind the imaginary line created by the Chairs. Before beginning the game you may try throwing the items as many times as you please. After starting you may follow the instructions below:
; Dodgeballs: you have 3 tosses each, 6 in total.
; CubeBlocks: you have only 1 toss each, 6 in total.
; AlarmClock: you have 1 toss.
; Book: you have 1 toss.
; If you toss the Dodgeballs or have more than 1 item in your Bin, you can cross the halfline and clear it preventing the overfill.
; SCORING: Each succesfull Dodgeball toss worth 8 points - 48 points total.
; Each CubeBlock worth 5 points - 30 points total.
; The AlarmClock worth 20 points.
; The Book worth 50 points.
; Total points: 148 points.
; DIFFICULTY: 1
(define (game 6123dcdd95e4f8afd71928a3-110) (:domain few-objects-room-v1)  ; 110
(:setup (and 
    (forall (?c - chair) (game-conserved (equal_x_position ?c door)))
    (exists (?h - hexagonal_bin) (game-conserved (and 
        (adjacent ?h south_west_corner)
        (faces ?h south_west_corner)
    )))
    (forall (?o - (either dodgeball cube_block alarm_clock book)) (game-optional (adjacent ?o desk)))
))
(:constraints (and 
    (forall (?o - (either dodgeball cube_block alarm_clock book)) (and 
        (preference throwFromBehindChairsInBin (exists (?h - hexagonal_bin)
            (then
                (once (and 
                    (agent_holds ?o)
                    (forall (?c - chair) (> (x_position agent) (x_position ?c)))
                ))
                (hold (and (not (agent_holds ?o)) (in_motion ?o)))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )   
        ))
        (preference throwAttempt 
            (then
                (once (and 
                    (agent_holds ?o)
                    (forall (?c - chair) (> (x_position agent) (x_position ?c)))
                ))
                (hold (and (not (agent_holds ?o)) (in_motion ?o)))
                (once (not (in_motion ?o)))
            )   
        )
    ))
))
(:terminal (or 
    (> (count-maximal-nonoverlapping throwAttempt:dodgeball) 3)
    (> (count-maximal-nonoverlapping throwAttempt:cube_block) 1)
    (> (count-maximal-nonoverlapping throwAttempt:book) 1)
    (> (count-maximal-nonoverlapping throwAttempt:alarm_clock) 1)
))
(:scoring maximize (+ 
    (* 8 (count-nonoverlapping throwFromBehindChairsInBin:dodgeball))
    (* 5 (count-nonoverlapping throwFromBehindChairsInBin:cube_block))
    (* 20 (count-nonoverlapping throwFromBehindChairsInBin:alarm_clock))
    (* 50 (count-nonoverlapping throwFromBehindChairsInBin:book))
)))

; 111 requires evaluation that one preference takes place before another preference is evaluated, and it's underconstrained

; 112 is definitely invalid and underdefined

; 6005e777d1d8768d5808b5fd-113 (few-objects-room-v1)
; SETUP: A ramp facing stacks of blocks creating columns leading to a bucket.
; GAMEPLAY: You roll a ball towards the ramp. The ball has to jump onto each column, bouncing to a next one and finally fall into the bucket.
; SCORING: You get a point if the ball gets into the bucket.
; DIFFICULTY: 3
(define (game 6005e777d1d8768d5808b5fd-113) (:domain few-objects-room-v1)  ; 113
(:setup (and 
    (exists (?h - hexagonal_bin ?c1 ?c2 ?c3 ?c4 - cube_block ?r - curved_wooden_ramp) (game-conserved (and 
        (adjacent_side ?h front ?c1)
        (adjacent ?c1 ?c3)
        (between ?h ?c1 ?c3)
        (on ?c1 ?c2)
        (on ?c3 ?c4)
        (adjacent_side ?r back ?c3)
        (between ?r ?c3 ?c1)
    )))
))
(:constraints (and 
    (preference ballThrownThroughRampAndBlocksToBin (exists (?b - ball ?r - curved_wooden_ramp ?h - hexagonal_bin ?c1 ?c2 - cube_block)
        (then
            (once (agent_holds ?b))
            (hold-while 
                (and (not (agent_holds ?b)) (in_motion ?b))
                (on ?r ?b)
                (on ?c1 ?b)
                (on ?c2 ?b)
            )
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
))
(:scoring maximize
    (count-nonoverlapping ballThrownThroughRampAndBlocksToBin)
))

; 61087e4fc006ee7d6be38641-114 (medium-objects-room-v1)
; SETUP: Move the dog bed into the center of the room
; GAMEPLAY: The game is to stack as many items as possible on the dog bed.  For an item to count, it cannot touch the ground or a wall, only the dog bed.
; SCORING: You get one point for every item that is on the dog bed.
; DIFFICULTY: 1
(define (game 61087e4fc006ee7d6be38641-114) (:domain medium-objects-room-v1)  ; 114
(:setup (and 
    (exists (?d - doggie_bed) (game-conserved (< (distance room_center ?d) 0.5)))
))
(:constraints (and 
    (preference objectInBuilding (exists (?o - game_object ?d - doggie_bed ?b - building)
        (at-end (and 
            (not (= ?o ?d))
            (in ?b ?d)
            (in ?b ?o)
            (on floor ?d)
            (not (on floor ?o))
            (not (exists (?w - wall) (touch ?w ?o))) 
        ))
    ))
))
(:scoring maximize
    (count-once-per-objects objectInBuilding)
))

; 5e606b1eaf84e83c728748d7-115 (medium-objects-room-v1)
; SETUP: Move the ramp to the room center. Place the chair in front of the ramp. Place the teddy bear on the chair. place the basket beyond the ramp and balls in front of the basket.
; GAMEPLAY: You jump the chair with the teddy bear and try to get the bear to land in the basket or hit balls placed near.
; SCORING: 1 point for each ball hit or 5 for getting the teddy in the basket.
; DIFFICULTY: 2
(define (game 5e606b1eaf84e83c728748d7-115) (:domain medium-objects-room-v1)  ; 115
(:setup (and 
    (exists (?c - chair ?r - triangular_ramp ?t - teddy_bear ?h - hexagonal_bin) (and 
        (game-conserved (and 
            (< (distance room_center ?r) 0.5)
            (adjacent_side ?r front ?c)
            (between ?h ?c ?r)
            (forall (?b - ball) (< (distance ?b ?h) 1))
        ))  
        (game-optional (and 
            (on ?c ?t)
        )) 
    ))
))
(:constraints (and 
    (preference teddyBearLandsInBin (exists (?t - teddy_bear ?h - hexagonal_bin ?c - chair)
        (then
            (once (on ?c ?t))
            (hold (agent_holds ?t))
            (hold (and (not (agent_holds ?t)) (in_motion ?t)))
            (once (and (not (in_motion ?t)) (in ?h ?t)))
        )   
    ))
    (preference teddyBearHitsBall (exists (?t - teddy_bear ?b - ball ?c - chair)
        (then
            (once (on ?c ?t))
            (hold (agent_holds ?t))
            (hold (and (not (agent_holds ?t)) (in_motion ?t)))
            (once (touch ?t ?b))
        )   
    ))
))
(:scoring maximize (+ 
    (* 5 (count-nonoverlapping teddyBearLandsInBin))
    (count-nonoverlapping teddyBearHitsBall)
)))

; 60bb404e01d599dfb1c3d71c-116 (medium-objects-room-v1)
; SETUP: Just move the basketball and dodgeball to middle of room, and put the bin on bed or desk
; GAMEPLAY: use the 2 balls and try to get them in bin
; SCORING: 4 attempts on each ball (dodgeball & basketball) and see  how many you can shoot in the bin
; DIFFICULTY: 3
(define (game 60bb404e01d599dfb1c3d71c-116) (:domain medium-objects-room-v1)  ; 116
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (or (on bed ?h) (on desk ?h))))
))
(:constraints (and 
    (forall (?b - (either basketball dodgeball)) (and 
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        (preference throwAttempt 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
    ))
))
(:terminal 
    (> (count-maximal-nonoverlapping throwAttempt) 4)
)
(:scoring maximize
    (count-nonoverlapping ballThrownToBin)
))

; 613e18e92e4ed15176362aa2-117 (medium-objects-room-v1)
; SETUP: To preprare the room for my game, you must put the wodden latter some steps back from the green garbage can, then crouch down and took the small red ball and start throwing it while you are crouch down,
; GAMEPLAY: To play my game you must decide how much intensity of force you must use in order to throw in the ball into de garbage can.
; SCORING: To score my game, 10 points will be if you scored the ball into the garbage can in the first attempt and wihtouht tocuhing the ground. You will get 7 points, if you throw in the ball into the garbage can between de second the fourth attempt and without touching the floor. Finally you will get 5 points if you just throw in the ball in the garbage can. You only have 10 attemps to score.
; DIFFICULTY: 1
(define (game 613e18e92e4ed15176362aa2-117) (:domain medium-objects-room-v1)  ; 117
(:setup (and 
    (exists (?h - hexagonal_bin ?r - triangular_ramp) (game-conserved (< (distance ?h ?r) 2)))
))
(:constraints (and 
    (preference redDodgeballThrownToBinWithoutTouchingFloor (exists (?h - hexagonal_bin ?r - red_dodgeball)
        (then 
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r) (not (touch floor ?r))))
            (once (and (not (in_motion ?r)) (in ?h ?r)))
        )
    ))
    (preference redDodgeballThrownToBin (exists (?h - hexagonal_bin ?r - red_dodgeball)
        (then 
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r)))
            (once (and (not (in_motion ?r)) (in ?h ?r)))
        )
    ))
    (preference throwAttempt (exists (?r - red_dodgeball)
        (then 
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r)))
            (once (not (in_motion ?r)))
        )
    ))
))
(:terminal (or 
    (>= (count-nonoverlapping throwAttempt) 10)
    (>= (count-once redDodgeballThrownToBinWithoutTouchingFloor) 1)
    (>= (count-once redDodgeballThrownToBin) 1)
))
(:scoring maximize (+ 
    (* 3
        (= (count-nonoverlapping throwAttempt) 1) 
        (count-once redDodgeballThrownToBinWithoutTouchingFloor)
    )
    (* 2
        (< (count-nonoverlapping throwAttempt) 5) 
        (count-once redDodgeballThrownToBinWithoutTouchingFloor)
    )
    (* 5 (count-once redDodgeballThrownToBin))
)))

; 5e73ded1027e893642055f86-118 (medium-objects-room-v1)
; SETUP: there is no need for a specific initial state
; GAMEPLAY: A game that could be played in this room is the "eco matchy matchy" game. During that game you have to put the various blocks on top/next to similar colored surfaces. For example, the green blocks could be put on the green part of the carpet or in the green basket. The light blue block could be put on the bed etc. Extra bonus are given to them that care to reduce the electricity consumption (switch off the lights). 
; SCORING: For ever block that is put on/next/in a matching object 5 points are given. When locating an object is harder (e.g. put a green block on the green basket, put a brown block on a brown drawer) 10 point are given. For the eco friendly participants , 15 points are given for each light that is turned off. Be careful, if you break anything (e.g. a computer, a window) you get a negative score: -10 for every broken object.
; DIFFICULTY: 2
(define (game 5e73ded1027e893642055f86-118) (:domain medium-objects-room-v1)  ; 118
(:constraints (and 
    (forall (?c - color) 
        (preference objectWithMatchingColor (exists (?o1 ?o2 - game_object)
            (at-end (and
                (= (color ?o1) (color ?o2))
                (= (color ?o1) ?c)
                (or 
                    (on ?o1 ?o2)
                    (adjacent ?o1 ?o2)
                    (in ?o1 ?o2)
                )
            ))
        ))
    )
    (preference itemsTurnedOff
        (exists (?o - (either main_light_switch lamp))
            (at-end 
                (not (toggled_on ?o))
            )
        )
    )
    (preference itemsBroken
        (exists (?o - game_object)
            (at-end 
                (broken ?o)
            )
        )
    )
))
(:scoring maximize (+ 
    (* 5 (count-once-per-objects objectWithMatchingColor))
    (* 5 (count-once-per-objects objectWithMatchingColor:green))
    (* 5 (count-once-per-objects objectWithMatchingColor:brown))
    (* 15 (count-once-per-objects itemsTurnedOff))
    (* (- 10 (count-once-per-objects itemsBroken)))
)))


; 23432t4543-120 (medium-objects-room-v1)
; SETUP: Put the bin on the bed
; GAMEPLAY: Throw the basketball into the bin
; SCORING: 1 point for each throw you make
; DIFFICULTY: 1
(
