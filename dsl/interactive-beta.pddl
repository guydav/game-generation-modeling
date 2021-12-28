













; 0 is a bit ambiguous, but I did my best

(define (game 6172feb1665491d1efbce164) (:domain medium-objects-room-v1)  ; 0
(:setup (and 
    (exists (?h - hexagonal_bin ?r - large_triangular_ramp)
        (game-conserved (< (distance ?h ?r) 1))
    )
))
(:constraints (and 
    (preference throwToRampToBin
        (exists (?b - ball ?r - large_triangular_ramp ?h - hexagonal_bin) 
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


(define (game 5f77754ba932fb2c4ba181d8) (:domain many-objects-room-v1)  ; 2
(:setup (and 
    (game-conserved (open top_drawer))
))
(:constraints (and 
    (preference throwToDrawerOrBin
        (exists (?b - (either dodgeball golfball) ?t - (either top_drawer hexagonal_bin))
            (then 
                (once (and (agent_holds ?b) (adjacent agent door)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (not (in_motion ?b)) (in ?t ?d)))
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
(define (game 614b603d4da88384282967a7) (:domain many-objects-room-v1)  ; 3
(:setup )
(:constraints (and 
    (forall (?b - building) 
        (preference blocktInTowerAtEnd (exists (?l - block)
            (at-end (in ?b ?l))
        ))
    )
))
(:scoring maximize (+
    (count-maximal-once-per-objects blocktInTowerAtEnd)
)))

; 4 is also woefully underconstrained

(define (game 5bc79f652885710001a0e82a) (:domain few-objects-room-v1)  ; 5
(:setup
)
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

(define (game 614dec67f6eb129c3a77defd) (:domain medium-objects-room-v1)  ; 6
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

; 7 is vastly under-constrained -- I could probably make some guesses but leaving alone

(define (game 615b40bb6cdb0f1f6f291f45) (:domain few-objects-room-v1)  ; 8
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
                    (agent_holds ?d1) 
                    (< (distance agent (side ?c front)) (distance agent (side ?c back)))
                ))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                ) 
                (once (and 
                    (not (in_motion ?d)) 
                    (< (distance ?d (side ?c back)) (distance ?d (side ?c front)))  
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
(define (game 615452aaabb932ada88ef3ca) (:domain many-objects-room-v1)  ; 9
(:setup (and 
    (exists (?h - hexagonal_bin)
        (or
            (game-conserved (on bed ?h))
            (exists (?w - wall) (game-conserved (adjacent ?w ?h)))
        )
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

(define (game 57aa430b4cda6e00018420e9) (:domain medium-objects-room-v1)  ; 10
(:setup 
)
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

(define (game 5d29412ab711e9001ab74ece) (:domain many-objects-room-v1)  ; 11
(:setup 
)
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
                    (not (exists (?o - game_object) (and (not (type ?o block)) (touch ?o ?b))))
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

(define (game 613bb29f16252362f4dc11a3) (:domain medium-objects-room-v1)  ; 12
(:setup (and 
    (exists (?h - hexagonal_bin)
        (game-conserved (< (distance ?h room_center) 1))
    )
))
(:constraints (and 
    (preference throwToRampToBin
        (exists (?c - curved_wooden_ramp ?d - dodgeball ?h - hexagonal_bin) 
            (then 
                (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?c ?d)
                ) 
                (once  (and (in ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
            )
        )
    )
))
(:scoring maximize
    (count-unique-positions throwToRampToBin)
))

(define (game 616e5ae706e970fe0aff99b6) (:domain many-objects-room-v1)  ; 13
(:setup (and 
    (exists (?h - hexagonal_bin ?r - large_triangular_ramp) (game-conserved 
        (and
            (< (distance ?h ?r) 1)
            (< (distance ?r room_center) 0.5)
        )
    ))
))
(:constraints (and 
    (forall (?d - (either dodgeball golfball))
        (preference throwToRampToBin
            (exists (?r - large_triangular_ramp ?h - hexagonal_bin) 
                (then 
                    (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
                    (hold-while 
                        (and (not (agent_holds ?d)) (in_motion ?d))
                        (touch ?r ?d)
                    ) 
                    (once (and (in ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
                )
            )
        )
    )
))
(:scoring maximize (+
    (* 6 (count-nonoverlapping throwToRampToBin:dodgeball))
    (* 3 (count-nonoverlapping throwToRampToBin:golfball))
)))

(define (game 609c15fd6888b88a23312c42) (:domain medium-objects-room-v1)  ; 14
(:setup
)
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

(define (game 5f5d6c3cbacc025bf0a03440) (:domain few-objects-room-v1)  ; 15
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


(define (game 616e4f7a16145200573161a6) (:domain few-objects-room-v1)  ; 16
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin ?b1 ?b2 ?b3 ?b4 - block) 
        (game-conserved (and
            (adjacent (side ?h front) (side ?c back))
            (on floor ?b1)
            (adjacent (side ?h left) ?b1)
            (on ?b1 ?b2)
            (on floor ?b3)
            (adjacent (side ?h right) ?b3)
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


(define (game 613e4bf960ca68f8de00e5e7) (:domain medium-objects-room-v1)  ; 17/18
(:setup
)
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

(define (game 60e93f64ec69ecdac3107555) (:domain medium-objects-room-v1)  ; 19
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


(define (game 5e2df2855e01ef3e5d01ab58) (:domain medium-objects-room-v1) ; 20
(:setup 
)
(:constraints (and 
    (forall (?b - building) (and  
        (preference blockInTowerAtEnd (exists (?l - block)
            (at-end
                (and 
                    (in building ?b ?l)
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
                    (in_building ?l1)
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

















