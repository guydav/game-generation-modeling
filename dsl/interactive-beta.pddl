













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

(define (game 5c79bc94d454af00160e2eee) (:domain few-objects-room-v1)  ; 21
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

(define (game 60d432ce6e413e7509dd4b78) (:domain medium-objects-room-v1)  ; 22
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (adjacent (bed ?h))))
    (forall (?b - ball) (game-optional (on rug ?b)))
    (not (exists (?g - game_object) (game-conserved (on desk ?g))))
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


(define (game 61267978e96853d3b974ca53) (:domain few-objects-room-v1)  ; 23
(:setup 
)
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
    (* (- 0.2) (count-nonoverlapping throwAttempt))
)))


(define (game 5996d2256b939900012d9f22) (:domain few-objects-room-v1)  ; 24
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

(define (game 606e4eb2a56685e5593304cd) (:domain few-objects-room-v1)  ; 27
(:setup (and 
    (forall (?d - (either dogeball cube_block)) (game-optional (not (exists (?s - shelf) (on ?s ?d)))))
    (game-optional (turned_on main_light_switch))
    (game-optional (turned_on desktop))
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
                (on ?s ?c)
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
        (exists (?o - (either main_light_switch))
            (at-end (and 
                (not (turned_on ?o))
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


(define (game 610aaf651f5e36d3a76b199f) (:domain few-objects-room-v1)  ; 28
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
(:terminal
    (>= (total-time) 180)
    (>= (total-score) 50)
)
(:scoring maximize (+
    (* 10 (count-nonoverlapping thrownBallReachesEnd))
    (* (- 5) (count-nonoverlapping thrownBallHitsBlock:red))
    (* (- 3) (count-nonoverlapping thrownBallHitsBlock:green))
    (* (- 3) (count-nonoverlapping thrownBallHitsBlock:pink))
    (* (- 1) (count-nonoverlapping thrownBallHitsBlock:yellow))
    (* (- 1) (count-nonoverlapping thrownBallHitsBlock:purple))
)))

(define (game 5bb511c6689fc5000149c703) (:domain few-objects-room-v1)  ; 29
(:setup
)
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


; 30 is rather underdetermined, I could try, but it would take some guesswork


(define (game 5b8c8e7d0c740e00019d55c3) (:domain few-objects-room-v1)  ; 31
(:setup (and 
    (exists (?h - hexagonal_bin) (and 
        (game-conserved (adjacent bed ?h))
        (forall (?b - cube_block) (adjacent ?h ?b))
    )
    (forall (?o - (either clock cellphone mug key_chain cd book ball))
        (game-optional (or 
            (on nightstand ?o)
            (on bed ?o)   
        ))
    )
))
(:constraints (and 
    (forall (?s - (either bed nightstand))
        (preference objectThrownFromRug
            (exists (?o - (either clock cellphone mug key_chain cd book ball) ?h - hexagonal_bin)
                (then
                    (once on ?s ?o)
                    (hold (and (agent_holds ?o) (on rug agent)))
                    (hold (and (not (agent_holds ?o)) (in_motion ?o))) 
                    (once (and (not (in_motion ?o)) (in ?h ?o)))
                )
            )
        )
    )
))
(:scoring maximize (+
    (count-nonoverlapping objectThrownFromRug:nightstand)
    (* 2 (count-nonoverlapping objectThrownFromRug:bed))
)))


(define (game 56cb8858edf8da000b6df354) (:domain many-objects-room-v1)  ; 32
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
            (hold-while (and (not (agent_holds ?d)) (in_motion ?d))
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
            (always
                (not (agent_holds ?d))
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


(define (game 614e1599db14d8f3a5c1486a) (:domain many-objects-room-v1)  ; 33
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


(define (game 615dd68523c38ecff40b29b4) (:domain few-objects-room-v1)  ; 35
(:setup
    
)
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


(define (game 5ef4c07dc8437809ba661613) (:domain few-objects-room-v1)  ; 36
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


(define (game 5fa45dc96da3af0b7dcba9a8) (:domain many-objects-room-v1)  ; 37
(:setup (and 

))
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

(define (game 616abb33ebe1d6112545f76d) (:domain medium-objects-room-v1)  ; 38
(:setup (and 

))
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


(define (game 614fb15adc48d3f9ffcadd41) (:domain many-objects-room-v1)  ; 39
(:setup 
)
(:constraints (and 
    (preference ballThrownToWallToAgent
        (exists (?b - ball ?w - wall) 
            (then
                (once (agent_holds ?b))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
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


(define (game 5c71bdec87f8cd0001b458f5) (:domain many-objects-room-v1)  ; 40
(:setup (and 
    (exists (?r - curved_wooden_ramp) (game-conserved (adjacent ?r rug)))
))
(:constraints (and 
    (forall (?c - color)
        (preference ballRolledOnRampToRug
            (exists (?b - beachball ?r - curved_wooden_ramp)
                (then 
                    (once (and (agent_holds ?b))
                    (hold-while 
                        (and (not (agent_holds ?b)) (in_motion ?d))
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
    (* 3 (count-nonoverlapping ballRolledOnRampToRug:blue))
    (* 4 (count-nonoverlapping ballRolledOnRampToRug:purple))
    (* (- 1) (count-nonoverlapping ballRolledOnRampToRug:white))
)))


(define (game 5f8d77f0b348950659f1919e) (:domain many-objects-room-v1)  ; 41
(:setup (and 
    (exists (?w1 ?w2 - wall) (and  
        (game-conserved (opposite ?w1 ?w2))
        (forall (?b - bridge_block) (game-conserved (and 
            (on floor ?b)
            (= (distance ?w1 ?b) (distance ?w2 ?b))    
        )))
        (forall (?g - game_object) (game-conserved (and 
            (not (type ?g bridge_block))
            (> (distance ?w1 ?b) (distance ?w2 ?b))
        )))
    )))
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


(define (game 5edc195a95d5090e1c3f91b2) (:domain few-objects-room-v1)  ; 42
(:setup (and 
    (exists (?h - hexagonal_bin) (and 
        (forall (?g - game_object) (game-optional (or
            (= ?h ?g)
            (> (distance ?h ?g) 1) 
        )))      
        (forall (?d - dodgeball) (game-optional (and
            (> (distance ?h ?g) 2) 
            (< (distance ?h ?g) 6) 
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
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
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


(define (game 617378aeffbba11d8971051c) (:domain medium-objects-room-v1)  ; 43
(:setup (and 
    (exists (?d - doggie_bed) (game-conserved (adjacent room_center ?d)))
))
(:constraints (and 
    (forall (?b - ball)
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
    )
))
(:scoring maximize (+
    (count-nonoverlapping throwBallToBin:basketball)
    (* 2 (count-nonoverlapping throwBallToBin:beachball))
    (* 3 (count-nonoverlapping throwBallToBin:dodgeball))
    (* 2 (count-nonoverlapping throwBallToBin:basketball))
    (* 3 (count-nonoverlapping throwBallToBin:beachball))
    (* 4 (count-nonoverlapping throwBallToBin:dodgeball)
)))

; 44 is another find the hidden object game

(define (game 60e7044ddc2523fab6cbc0cd) (:domain many-objects-room-v1)  ; 45
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
    (forall (?b - (either golfball dodgeball)) 
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
                    (and (in_motion ?b) (not (agent_holds ?b))
                    (touch ?b ?t)
                ))
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
    )
))
(:terminal (or 
    (> (count-maximal-nonoverlapping throwAttempt) 1)
    (>= (count-once-per-objects throwAttempt) 6)
))
(:scoring maximize (+
    (count-once-per-objects throwKnocksOverBear:dodgeball)
    (* 2 (count-once-per-objects throwKnocksOverBear:golfball))   
)))

(define (game 5d5b0dd7c032a2001ad7cf5d) (:domain few-objects-room-v1)  ; 46
(:setup (and 
    (exists (?c - curved_wooden_ramp) (game_conserved (and
        (< (distance ?c room_center) 3)  
    )))
))
(:constraints (and 
    (preference ballThrownToRampToBed (exists (?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds purple_dodgeball) (faces agent ?c)))
            (hold-while
                (and (in_motion purple_dodgeball) (not (agent_holds purple_dodgeball)))
                (touch purple_dodgeball ?c)
            )
            (once (and (not (in_motion purple_dodgeball)) (on bed purple_dodgeball)))
        )
    ))
    (preference ballThrownHitsAgent (exists (?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds purple_dodgeball) (faces agent ?c)))
            (hold-while
                (and (in_motion purple_dodgeball) (not (agent_holds purple_dodgeball)))
                (touch purple_dodgeball ?c)
            )
            (once (and (touch purple_dodgeball agent) (not (agent_holds purple_dodgeball))))
        )
    ))
))
(:scoring maximize (+ 
    (count-nonoverlapping ballThrownToRampToBed)
    (* (- 1) (count-nonoverlapping ballThrownHitsAgent))
)))


(define (game 5d470786da637a00014ba26f) (:domain many-objects-room-v1)  ; 47
(:setup (and 

))
(:constraints (and 
    (forall (?c - color) 
        (preference beachballBouncedOffRamp
            (exists (?b - beachball ?r - green_small_ramp)
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

;48 is complicated as fuck -- leaving for last

(define (game 60ddfb3db6a71ad9ba75e387) (:domain many-objects-room-v1)  ; 49
(:setup (and 
    (game-conserved (< (distance green_golfball door) 0.5))
    (forall (?d - dodgeball) (game-optional (< distance green_golfball ?d)) 1)
))
(:constraints (and 
    (forall (?d - dodgeball) 
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
    )
))
(:terminal (or 
    (> (count-maximal-nonoverlapping throwAttemptFromDoor) 1)
    (>= (count-once-per-objects throwAttemptFromDoor) 3)
))
(:scoring maximize
    (* 10 (count-once-per-objects dodgeballThrownToBin))
))

(define (game 5f3aee04e30eac7cb73b416e) (:domain medium-objects-room-v1)  ; 50
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (< (distance room_center ?h) 1)))
))
(:constraints (and 
    (preference gameObjectToBin (exists (?g - game_object ?h - hexagonal_bin)) 
        (then 
            (once (not (agent_holds ?g)))
            (hold (or (agent_holds ?g) (in_motion ?g)))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    )
))
(:scoring maximize
    (count-once-per-objects gameObjectToBin)
))

(define (game 5ff4a242cbe069bc27d9278b) (:domain few-objects-room-v1)  ; 51
(:setup
)
(:constraints (and 
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring maximize
    (count-nonoverlapping throwToBin)  
))


(define (game 602d84f17cdd707e9caed37a) (:domain few-objects-room-v1)  ; 52
(:setup 
)
(:constraints (and 
    (preference blockFromRugToDesk (exists (?c - cube_block ) 
        (then 
            (once (and (on rug agent) (agent_holds ?c)))
            (hold (and 
                (on rug agent)
                (in_motion ?c)
                (not (agent_holds ?c))
                (not (exists (?o - either lamp desktop laptop)) (or (is_broken ?o) (in_motion ?o)))
            ))
            (once (and (on rug agent) (on desk ?c) (not (in_motion ?c))))
        )
    ))
))
(:scoring maximize
    (count-once-per-objects blockFromRugToDesk)
))


(define (game 5f0cc31363e0816c1b0db7e1) (:domain few-objects-room-v1)  ; 53
(:setup 
)
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
        (exists (?o - (either cellphone key_chain mug credit_card cd watch) ?d - drawer)
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


