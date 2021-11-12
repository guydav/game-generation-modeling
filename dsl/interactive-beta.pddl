(define (game 5e2df2855e01ef3e5d01ab58) (:domain medium-objects-room-v1) ; 0
(:setup 
)
(:constraints (and 
    (preference baseBlockInTowerAtEnd (exists (?b1 - block)
        (at-end
            (and 
                (in_building ?b1)
                (not (exists (?b2 - block) (on ?b1 ?b2)))
            )
        )
    ))
    (preference blockOnBlockInTowerAtEnd (exists (?b1 - block)
        (at-end
            (and 
                (in_building ?b1)
                (exists (?b2 - block) (on ?b1 ?b2))
            )
        )
    )) 
    (preference blockInTowerKnockedByDodgeball (exists (?b - block ?d - dogeball)
        (then
            (once (and (in_building ?b) (agent_holds ?d)))
            (hold (and (in_building ?b) (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (in_building ?b) (touch ?d ?b)))
            (hold (in_motion ?b))
            (once (not (in_motion ?b)))
        )
    ))
    (preference towerFallsWhileBuilding (exists (?b1 ?b2 - block))
        (throwBetweenBlocksToBin
            (once (and (in_building ?b1) (agent_holds ?b2)))
            (hold-while 
                (and
                    (not (agent_holds ?b1)) 
                    (in_building ?b1)
                    (or 
                        (agent_holds ?b2) 
                        (and (not agent_holds ?b2) (in_motion ?b2))
                    )
                )
                (touch ?b1 ?b2)
            )
            (once (on floor ?b1))
        )
    )
))
(:scoring maximize (+ 
    (count-once-per-objects baseBlockInTowerAtEnd)
    (count-once-per-objects blockOnBlockInTowerAtEnd)
    (* 2 (count-once-per-objects blockInTowerKnockedByDodgeball))
    (* (- 1) (count-nonoverlapping towerFallsWhileBuilding))
)))


(define (game 60e93f64ec69ecdac3107555) (:domain medium-objects-room-v1)  ; 1
(:setup (and
    (forall (?b - (either basketball beachball dodgeball))
        (game-optional (< (distance ?b door) 1))
    )
))
(:constraints (and 
    (preference beachballThrownFromDoorToDoggieBed
        (exists (?d - doggie_bed ?b - beachball) 
            (then 
                (once (and (agent_holds ?b) (< (distance agent door) 1)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (on ?d ?b) (not (in_motion ?b))))
            )
        )
    )
    ; TODO: several more of these -- would immensly benefit from the forall over preferences 
    ; TODO: which maybe doesn't require any unique syntax in prefernece specification, only in scoring
    ; TODO: Basically, *only* if you have a variable defined with an (either ...), it might make sense to refer
    ; TODO: to which type was used to satisfy the preference, and score accordingly
    ; TODO: except the next game made me think about cases where there's an implicit either --
    ; TODO: `block` standing for any of the blocks, or `dodgeball` for any color of dodgeball.
    ; TODO: more to think about
)
(:scoring maximize (+ 
    ; TODO: use as a test case for the forall over preference types
)))

(define (game 60e93f64ec69ecdac3107555) (:domain medium-objects-room-v1)  ; 2 
(:setup
)
(:constraints (and 
    (preference castleBuilt (?b - bridge_block ?f - flat_block ?t - tall_cylindrical_block ?c - cube_block ?p - pyramid_block)
        (at-end
            (and 
                (on ?b ?f)
                (on ?f ?t)
                (on ?t ?c)
                (on ?c ?p)
            )
        )
    )
)(:scoring maximize (+ 
    ; TODO: use as another test case for the forall over preference types
)))

; 3 is a dup of 2

(define (game 616e4f7a16145200573161a6) (:domain few-objects-room-v1)  ; 4 
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin ?b1 ?b2 ?b3 ?b4 - block) 
        (game-conserved (and
            (adjacent (side ?h front) (side ?c back)))
            (on floor ?b1)
            (adjacent (side ?h left) ?b1)
            (on ?b1 ?b2)
            (on floor ?b3)
            (adjacent (side ?h right) ?b3)
            (on ?b3 ?b4)
        )
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
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
            )
        ) 
    )
)(:scoring maximize (count-nonoverlapping rollBallToBin)
))

(define (game 5f5d6c3cbacc025bf0a03440) (:domain few-objects-room-v1)  ; 5
(:setup (and
    (exists ?h - hexagonal_bin) 
        (game-conserved (and
            (adjacent ?h bed)
            (object_orientation ?h upside_down)
        )
        (game-optional (forall (?b - cube_block) (or 
            (on ?h ?b)
            (exists (?b2 - cube_block) (on ?b ?b2))   
        )))
    )
))
(:constraints (and 
    (preference blockInTowerKnockedByDodgeball (exists (?b - cube_block ?d - dogeball ?h - hexagonal_bin ?c - chair)
        (then
            (once (and 
                (agent_holds ?d)
                (adjacent agent ?c)
                (or 
                    (on ?h ?b)
                    (exists (?b2 - cube_block) (on ?b2 ?b))
                )    
            ))
            (hold-while (and (not (agent_holds ?d)) (in_motion ?d))
                (or 
                    (touch ?b ?d)
                    (exists (?b2 - cube_block) (touch ?b2 ?b))
                )
                (in_motion ?b)
            )
            (once (not (in_motion ?b)))
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


(define (game 609c15fd6888b88a23312c42) (:domain medium-objects-room-v1)  ; 6
(:setup
)
(:constraints (and 
    (preference throwInBin
        (exists (?b - ball ?h - hexagonal_bin)
            (then 
                (once (and (on rug agent) (agent_holds ?b)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)) (on ?h ?b))
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

(define (game 616e5ae706e970fe0aff99b6) (:domain many-objects-room-v1)  ; 7
(:setup (and 
    (exists (?h - hexagonal_bin ?r - large_triangular_ramp) (game-conserved 
        (and
            (< (distance ?h ?r) 1)
            (< (distance ?r room_center) 0.5)
        )
    ))
))
(:constraints (and 
    (preference throwToRampToBin
        (exists (?d - dodgeball ?r - large_triangular_ramp ?h - hexagonal_bin) 
            (then 
                (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?r ?d)
                ) 
                (once  (and (on ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
            )
        )
    )
))
(:scoring maximize
    ; TODO another place to try out the counting per object type (3 for dodgeballs, 6 for golfballs)
))

; 8 requires quantifying based on position -- something like

(define (game 613bb29f16252362f4dc11a3) (:domain medium-objects-room-v1)  ; 8
(:setup (and 
    (exists (?h - hexagonal_bin)
        (game-conserved (< (distance ?h room_center) 1))
    )
))
(:constraints (and 
    (preference throwToRampToBin
        (exists (?d - dodgeball ?c - curved_wooden_ramp ?h - hexagonal_bin) 
            (then 
                (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?c ?d)
                ) 
                (once  (and (on ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
            )
        )
    )
))
(:scoring maximize
    (count-unique-positions throwToRampToBin (?c - curved_wooden_ramp))  ; repeat the entire variable def? Or just the variable?
))


(define (game 5d29412ab711e9001ab74ece) (:domain many-objects-room-v1)  ; 9
(:setup 
)
(:constraints (and 
    (preference baseBlockInTowerAtEnd (exists (?b1 - block)
        (at-end
            (and 
                (in_building ?b1)
                (not (exists (?b2 - block) (on ?b1 ?b2)))
            )
        )
    ))
    (preference blockOnBlockInTowerAtEnd (exists (?b1 - block)
        (at-end
            (and 
                (in_building ?b1)
                (exists (?b2 - block) (on ?b1 ?b2))
                (exists (?b2 - block) (on ?b2 ?b1))
            )
        )
    )) 
    (preference pyramidBlockAtopTowerAtEnd (exists (?p - pyramid_block)
        (at-end
            (and 
                (in_building ?p)
                (exists (?b - block) (on ?b ?p))
                (not (exists (?b - block) (on ?p ?b)))
            )
        )
    )) 
))
(:scoring maximize (* 
    (count-once pyramidBlockAtopTowerAtEnd)
    (+ 
        (count-once pyramidBlockAtopTowerAtEnd)
        (count-once baseBlockInTowerAtEnd)
        (count-once-per-objects blockOnBlockInTowerAtEnd)   
    )     
)))
