; create tunnels with the brigde blocks and throw the balls
; "1 point for teach"

(define (game medium-objects-2) (:domain medium-objects-room-v1)
(:setup (and
    (forall (?f - flat_block) (game-conserved (on floor ?f)))
    (forall (?b - bridge_block) 
        (exists (?f - flat_block) 
            (game-conserved (on ?f ?b))
        )
    )
))
(:constraints (and 
    (preference throwBallUnderBridge
        (exists (?b - basketball ?bb - bridge_block)  
            (then 
                ; ball starts in hand, not under the bridge
                (once (and (agent_holds ?b) (not (under ?bb ?b))))
                ; neither ball nor block in hand until...
                (hold-while 
                    (and (in_motion ?b) (not (agent_holds ?b)) (not (agent_holds ?bb)))
                    (under ?bb ?b)
                )     
                ; the ball is under the bridge and then again not under the bridge
                (once (not (under ?bb ?b))) 
            ) 
        ) 
    ) 
))
(:scoring maximize (count-nonoverlapping throwBallUnderBridge)) 
)


(define (game medium-objects-3) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?s - shelf ?h - hexagonal_bin) 
        (and
            (game-conserved (on ?s ?h))
            (forall (?s2 - shelf) (game-conserved (>= (distance ?s desk) (distance ?s2 desk))))
        )
    )
))
(:constraints (and 
    (preference throwBallFromChairToBin
        (exists (?b - basketball ?c - chair ?h - hexagonal_bin) 
            (then
                ; ball starts in hand, with the agent on the chair, near the desk
                (once (and (agent_holds ?b) (on ?c agent) (adjacent ?c desk) (agent_perspective looking_upside_down)))
                ; ball not in hand until...
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                ; the ball is in the bin
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )  
    )
)) 
(:scoring maximize (count-nonoverlapping throwBallFromChairToBin))
)


(define (game medium-objects-4) (:domain medium-objects-room-v1)
(:setup (and
    (forall (?b - block) (game-optional (on floor ?b)))
))
(:constraints (and
    ; Here we have the preference before the quantifier, to count it at most once
    (preference blockOnFloor (exists (?b - block) 
        (then 
            (hold
                (and
                    (on floor ?b)
                    (in_building ?b)
                )
            )
            (once (building_fell))
        )
    ))
    ; Here we have the quantifier before, to count how many times it happens 
    (preference blockOnBlock (exists (?b - block ?b2 - block)
        (then
            (hold
                (and 
                    ; both blocks are in the tower
                    (in_building ?b)
                    (in_building ?b2)
                    ; this new block ?b is on top of the second block ?b2
                    (on ?b ?b2) ; an object cannot be on itself, so this fails if ?b = ?b2
                )
            )
            ; until the tower falls
            (once (building_fell))
        )
    ))
    (preference blockFellNear (exists (?b - block) 
        (then
            ; block is in the towr until
            (hold (in_building ?b)
            ; starting with the building falling
            (once (building_fell))
            ; block is falling without agent moving it until -- this only works if the blocks start moving the state after the previous state happens
            (hold (and (not (agent_holds ?b) (in_motion ?b)))) 
            ; it settles near the tower
            (once (<= (distance building ?b) 0.1)) 
        )
    )) 
))
(:scoring maximize (+
    (count-once blockOnFloor)
    (count-once-per-objects blockOnBlock)
    (- (count-once-per-objects blockFellNear))
))
)

(define (game medium-objects-5) (:domain medium-objects-room-v1)
(:setup )
(:constraints (and
    ; Count how many objects are part of the tower
    (preference objectInTower (exists (?o - game_object)
        (then
            (once (agent_holds ?b))
            (hold (in_building ?o))
            (once (building_fell))
        )
    ))
))
(:scoring scoring (/ (* 100 (max_building_height)) (count-once-per-objects objectInTower))
)
)

;6 is invalid
 
(define (game medium-objects-7) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?h - hexagonal_bin) (exists (?d - doggie_bed) (exists (?p - pillow)
        (game-conserved (and
                (<= (distance ?h ?d) 1)
                (<= (distance ?h ?p) 1)
                (<= (distance ?p ?d) 1)
                (< (distance agent ?h) (distance agent ?d))
                (< (distance agent ?h) (distance agent ?p))
                (< (distance agent ?d) (distance agent ?p))
        ))
    )))
))
(:constraints (and
    (preference beachballToHexagonalBin
        (exists (?b - beachball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference beachballToDoggieBed
        (exists (?b - beachball ?d - doggie_bed)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?d ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference beachballToPillow
        (exists (?b - beachball ?p - pillow) 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?p ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference dodgeballToHexagonalBin
        (exists (?b - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference dodgeballToDoggieBed
        (exists (?b - dodgeball ?d - doggie_bed)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?d ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference dodgeballToPillow
        (exists (?b - dodgeball ?p - pillow) 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?p ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference basketballToHexagonalBin
        (exists (?b - basketball ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference basketballToDoggieBed
        (exists (?b - basketball ?d - doggie_bed)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?d ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference basketballToPillow
        (exists (?b - basketball ?p - pillow)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?p ?b) (not (in_motion ?b))))
            )
        )
    )
))
(:scoring maximize (+ 
    (* 3 (count-nonoverlapping beachballToHexagonalBin))
    (* 5 (count-nonoverlapping beachballToDoggieBed))
    (* 7 (count-nonoverlapping beachballToPillow))
    (* 6 (count-nonoverlapping dodgeballToHexagonalBin))
    (* 8 (count-nonoverlapping dodgeballToDoggieBed))
    (* 10 (count-nonoverlapping dodgeballToPillow))
    (* 9 (count-nonoverlapping basketballToHexagonalBin))
    (* 11 (count-nonoverlapping basketballToDoggieBed))
    (* 13 (count-nonoverlapping basketballToPillow))
)))


(define (game medium-objects-8) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?t1 ?t2 - tall_cylindrical_block ?tb - teddy_bear)
        (game-conserved (and
            (<= (distance ?t1 ?t2) 2)
            (= (distance ?tb ?t1) (distance ?tb ?t2))
        ))
    )
))
(:constraints (and 
    (preference throwBetweenBlocksToBear
        (exists (?b - basketball ?t1 - tall_cylindrical_block ?t2 - tall_cylindrical_block ?tb - teddy_bear)
            (then 
                (once (agent_holds ?b))
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (between ?t1 ?b ?t2)
                )
                (once (touch ?b ?tb))
            )
        ) 
    )
    (preference thrownBallHitBlock
        (exists (?b - basketball ?t - tall_cylindrical_block) 
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))block
                (once (touch ?b ?t)) 
            )
        ) 
    )
    (preference throwAttempt
        (exists (?b - basketball)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 5)
)
(:scoring maximize (+
    (* 15 (count-nonoverlapping throwBetweenBlocksToBear))
    (* (- 5) (count-nonoverlapping thrownBallHitBlock))
))


(define (game medium-objects-9) (:domain medium-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference thrownObjectKnocksDesktop
        (exists (?o - (either pillow beachball dodgeball) ?d - desktop)
            (then
                ; starts with agent holding the desktop
                (once (agent_holds ?o))
                (hold-while
                    ; while the object is being thrown and the agent is touching neither the object nor the desktop
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?d)))
                    ; the thrown object hits the desktop
                    (touch ?o ?d)
                )
                ; eventually knocking it off the desk
                (once (and (not (on desk ?d)) (not (in_motion ?d)))) 
            )
        ) 
    )
    (preference thrownObjectKnocksDeskLamp
        (exists (?o - (either pillow beachball dodgeball) ?d - desk_lamp)
            (then
                ; starts with agent holding the desktop
                (once (agent_holds ?o))
                (hold-while
                    ; while the object is being thrown and the agent is touching neither the object nor the desk lamp
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?d)))
                    ; the thrown object hits the desk lamp
                    (touch ?o ?d)
                )
                ; eventually knocking it off the desk
                (once (and (not (on desk ?d)) (not (in_motion ?d)))) 
            )
        ) 
    )
    (preference thrownObjectKnocksCD
        (exists (?o - (either pillow beachball dodgeball) ?c - cd)
            (then
                ; starts with agent holding the desktop
                (once (agent_holds ?o))
                (hold-while
                    ; while the object is being thrown and the agent is touching neither the object nor the cd
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?c)))
                    ; the thrown object hits the cd
                    (touch ?o ?c)
                )
                ; eventually knocking it off the desk
                (once (and (not (on desk ?c)) (not (in_motion ?c)))) 
            )
        ) 
    ))
))
(:scoring maximize (+
    (* 5 (count-nonoverlapping thrownObjectKnocksDesktop))
    (* 10 (count-nonoverlapping thrownObjectKnocksDeskLamp))
    (* 15 (count-nonoverlapping thrownObjectKnocksCD))
))


(define (game medium-objects-10) (:domain medium-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference throwBallWithEyesClosed
        (exists (?b - basketball ?h - hexagonal_bin) 
            (then
                ; ball starts in hand, with the agent on the chair, near the desk
                (once (and (agent_holds ?b) (agent_perspective eyes_closed)))
                ; ball not in hand and in motion until...
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                ; the ball is in the bin
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        ) 
    )
))
(:terminal
    (>= (total-score) 100)
)
(:scoring (* 5(count-nonoverlapping throwBallFromChairToBin)))
)

(define (game medium-objects-11) (:domain medium-objects-room-v1)
(:setup )
(:constraints (and
    (preference correctColorBlock (exists (?b - block) 
        (at-end
            (and 
                (in_building ?b))
                (or
                    (exists (?b2 - bridge_block) (and (= ?b ?b2) (object_color ?b green)))
                    (exists (?b2 - pyramid_block) (and (= ?b ?b2) (object_color ?b red))
                    (exists (?b2 - short_cylindrical_block) (and (= ?b ?b2) (or (object_color ?b green) (object_color ?b blue) )))
                    (exists (?b2 - flat_block) (and (= ?b ?b2) (object_color ?b yellow))))
                    (exists (?b2 - cube_block) (and (= ?b ?b2) (object_color ?b blue)))
                )
        )
    ))
))
(:scoring maximize (* (2 (count-once-per-objects correctColorBlock)))
))


(define (game medium-objects-12) (:domain medium-objects-room-v1)
(:setup 
)
(:constraints (and 
    (preference throwDodgeballToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (> (distance agent ?h) 5)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?d) (not (in_motion ?d))))
            )
        ) 
    )
    (preference throwBeachballToBin
        (exists (?b - beachball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?b) (> (distance agent ?h) 5)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        ) 
    )
    (preference throwBasketballToBin
        (exists (?b - basketball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?b) (> (distance agent ?h) 5)))
                (hold (and (and (not (agent_holds ?b)) (in_motion ?b))))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        ) 
    )
))
(:scoring maximize (+
    (* 1 (count-nonoverlapping throwDodgeballToBin))
    (* 2 (count-nonoverlapping throwBeachballToBin))
    (* 3 (count-nonoverlapping throwBasketballToBin))
)))

; 13 is effectively invalid -- no real scoring + references to multiple agents


(define (game medium-objects-14) (:domain medium-objects-room-v1)
(:setup 
    (exists (?w - wall ?h - hexagonal_bin) 
        (game-conserved (adjacent ?w ?h))
    )
)
(:constraints (and 
    (preference throwDodgeballToBin
        (exists (?d - dodgeball ?h - hexagonal_bin ?w1 ?w2 - wall)
            (then 
                (once (and (agent_holds ?d) (adjacent ?h ?w1) (opposite ?w1 ?w2) (adjacent agent ?w2)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?d) (not (in_motion ?d))))
            )
        ) 
    )
))
(:scoring maximize (count-nonoverlapping throwDodgeballToBin)
))


(define (game medium-objects-15) (:domain medium-objects-room-v1)
(:setup 
    (exists (?r - large_triangular_ramp ?h - hexagonal_bin) 
        (game-conserved (adjacent ?r ?h))
    )
)
(:constraints (and 
    (preference rollBallToBin
        (exists (?d - dodgeball ?r - large_triangular_ramp ?h - hexagonal_bin) 
            (then 
                (once-measure (agent_holds ?d) (distance agent ?h) )
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) 
                    (on ?r ?d) 
                )
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
            )
        )
    )
))
(:scoring maximize (* 
    ; "the points increase by one with each step back" - n * n-1
    (count-increasing-measure rollBallToBin)
    (+ (count-increasing-measure rollBallToBin) (- 1))
)))


(define (game medium-objects-16) (:domain medium-objects-room-v1)
(:setup  
    (forall (?b - (either cube_block flat_block))
        (exists (?b2 - (either cube_block flat_block) ?h - hexagonal_bin) 
            (game-optional (and 
                (not (= ?b ?b2))
                (or 
                    (on ?b ?b2)
                    (on ?b2 ?b)
                    (adjacent ?b ?b2)
                )
                (< (distance ?h ?b) 1.5)
            ))
        )
    )
)
(:constraints (and 
    (preference throwAttempt
        (exists (?b - (either dodgeball basketball))
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
    (preference allBlocksHit
        ; see notes on above
        (forall (?bl - (either cube_block flat_block))
            (exists (?b - (either dodgeball basketball))
                (then
                    (hold-while (not (touch agent ?bl))
                        (in_motion ?b)
                        (touch ?bl ?b)
                        (in_motion ?bl)
                        (not (in_motion ?bl))
                    )
                )
            )
        )
    )
    (preference throwInBin
        (exists (?b - (either dodgeball basketball) ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    
))
(:terminal (or
    (>= (count-nonoverlapping throwAttempt) 3)
    (and
        (> (count-once allBlocksHit) 0)
        (> (count-once throwInBin) 0)
    )
))
(:scoring maximize (+ 
    (* 100 (count-once allBlocksHit) (count-once throwInBin) (= (count-nonoverlapping throwAttempt) 2))
    (* 75 (count-once allBlocksHit) (count-once throwInBin) (= (count-nonoverlapping throwAttempt) 3))
    (* 15 (count-once allBlocksHit) (= (count-once throwInBin) 0) (= (count-nonoverlapping throwAttempt) 3))
)))



(define (game medium-objects-17) (:domain few-objects-room-v1)
(:setup 
)
(:constraints (and 
    (preference throwToWallToBin
        (exists (?d - dodgeball ?w - wall ?h - hexagonal_bin) 
            (then 
                (once (and (agent_holds ?d) (on bed agent)))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?w ?d)
                ) 
                (once (and (on ?h ?d) (not (in_motion ?d))))
            )
        )
    )
    (preference throwMisses
        (exists (?d - dodgeball) 
            (then 
                (once (and (agent_holds ?d) (on bed agent)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and 
                    (not (in_motion ?d))
                    (forall (?h - hexagonal_bin (not (on ?h ?d))))
                ))
            )
        )
    )
))
(:terminal
    (> (count-once throwMissesBin) 0)
)
(:scoring maximize (count-nonoverlapping throwToWallToBin)) 
)


(define (game medium-objects-18) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (object_orientation ?h upside_down)))
))
(:constraints (and
    (preference objectInTower (exists (?o - game_object)
        (at-end
            (and
                (in_building ?o)
                (or
                    (exists (?h - hexagonal_bin) 
                        (and
                            (object_orientation ?h upside_down)
                            (on ?h ?o)
                        )
                    )
                    (exists (?o2 - game-object) 
                        (and
                            (in_building ?o2)
                            (not (= ?o ?o2))
                            (on ?o2 ?o)
                        )
                    )
                )
            )
        )
    ))
))
(:scoring maximize (+
    (* 10 (count-once-per-objects objectInTower))
    (* 100 (building_height))
)))

; 19 is invalid


(define (game medium-objects-20) (:domain medium-objects-room-v1)
(:setup 
)
(:constraints (and 
    (preference throwBasketballToBin
        (exists (?b - basketball ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?b) (> (distance agent ?h) 5)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        ) 
    )
    (preference throwAttempt
        (exists (?b - basketball)
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
(:scoring maximize (count-nonoverlapping throwBasketballToBin)
))


(define (game medium-objects-21) (:domain medium-objects-room-v1)
(:setup 
    (exists (?b1 ?b2 - cube_block ?c1 ?c2 - short_cylindrical_block ?h - hexagonal_bin)
        (game-conserved (and
            (= 2 (distance ?b1 ?b2))
            (= 2 (distance ?c1 ?c2))
            (= 2 (distance ?b1 ?c1))
            (= 2 (distance ?b2 ?c2))
            (= 0.5 (distance ?h south_wall))  ; assuming it's the south one, as it looks like a sunny morning with sun from east
        ))
    )
)
(:constraints (and 
    (preference throwBasketballToBinAfterDribbling
        (exists (?b - basketball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (forall-sequence (?c - (either cube_block short_cylindrical_block))
                    (hold (agent_dribbles ?b))
                    (once (agent_circled_around ?c))
                    (any)  ; to give a gap to get to the next block
                )
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (on ?h ?b) (not (in_motion ?b)))
            )
        ) 
    )
))
(:terminal
    (>= (total-time) 120)
)
(:scoring maximize (count-nonoverlapping throwBasketballToBinAfterDribbling)
))

; 22 is invalid


(define (game medium-objects-23) (:domain medium-objects-room-v1)
(:setup (and
    (forall (?b - bridge_block) 
        (game-conserved (and (on floor ?b) (object_orientation ?b upright)))
    )
    (exists (?r - large_triangular_ramp ?b - bridge_block) 
        (game-conserved (= 0.67 (distance ?r ?b)) )
    )
    (forall (?o - (either cellphone laptop doggie_bed mug))
        (exists (?b - bridge_block ?r - large_triangular_ramp)
            (game-optional (and 
                (between ?o ?b ?r)
                (= 0.67 (distance ?o ?b))
            ))
        )
    )
))
(:constraints (and 
    (preference rollBallToObjects
        (exists (?b - (either baseketball dodgeball beachball) ?r - large_triangular_ramp
                ?o - (either cellphone laptop doggie_bed mug))  
            (then 
                (once (agent_holds ?b))
                (hold-while 
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (on ?r ?b)
                )     
                (once (touch ?o ?b))  
            )
        ) 
    ) 
    (preference rollBlockToObjects
        (exists (?c - tall_cylindrical_block ?r - large_triangular_ramp
                ?o - (either cellphone laptop doggie_bed mug))  
            (then 
                (once (agent_holds ?c))
                (hold-while 
                    (and (in_motion ?c) (not (agent_holds ?c)))
                    (on ?r ?c)
                )     
                (once (touch ?o ?c))  
            )
        ) 
    )
    (preference rollCDToObjects
        (exists (?c - cd ?r - large_triangular_ramp
                ?o - (either cellphone laptop doggie_bed mug))  
            (then 
                (once (agent_holds ?c))
                (hold-while 
                    (and (in_motion ?c) (not (agent_holds ?c)))
                    (on ?r ?c)
                )     
                (once (touch ?o ?c))  
            )
        ) 
    )
    (preference rolledObjectHitsBlock
        (exists (?b - (either baseketball dodgeball beachball tall_cylindrical_block cd) 
                ?r - large_triangular_ramp ?bb - bridge_block)  
            (then 
                (once (agent_holds ?b))
                (hold-while 
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (on ?r ?b)
                )     
                (once (touch ?bb ?b))  
            )
        ) 
    ) 
) )
(:scoring maximize (+
    (* 5 (count-nonoverlapping rollBallToObjects)) 
    (* 8 (count-nonoverlapping rollBlockToObjects))
    (* 10 (count-nonoverlapping rollCDToObjects))
    (* (- 5) (count-nonoverlapping rolledObjectHitsBlock))
)))


(define (game medium-objects-24) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?r - large_triangular_ramp) (forall (?b - block)
        (and 
            (exists (?b2 - block)
                (game-optional (and 
                    (not (= ?b ?b2))
                    (or 
                        (on ?b2 ?b)
                        (adjacent ?b2 ?b)
                    )
                ))
            )
            (forall (?b2 - block) 
                (game-optional (and 
                    (not (= ?b ?b2))
                    (< (distance ?b ?b2) 0.5)
                ))
            )
            (game-optional (> (distance ?r ?b) 0.75))
            (game-optional (< (distance ?r ?b) 1.25))
        )
    ))
))
(:constraints (and 
    (preference rollBallToWall
        (exists (?d - dodgeball ?r - large_triangular_ramp ?b - block)
            (then 
                (once (agent_holds ?d))
                (hold-while 
                    (and (in_motion ?d) (not (agent_holds ?d)))
                    (on ?r ?d)
                    (once (touch ?b ?d))
                    (in_motion ?b)
                )     
            )
        ) 
    ) 
) )
(:scoring maximize (count-once-per-objects rollBallToWall)  
))

; 25 requires counting how many times something happens in a preference

; 26 is invalid

(define (game medium-objects-27) (:domain medium-objects-room-v1)

(:setup (and
    (exists (?h - hexagonal_bin ?d - doggie_bed ?c1 ?c2 - tall_cylindrical_block
            ?p1 ?p2 - pyramid_block ?r - large_triangular_ramp) 
        (game-conserved (and
            (object_orientation ?h upside_down)
            (= 0.67 (distance ?h ?d))
            (between ?h ?c1 ?d)
            (between ?h ?c2 ?d)
            (on ?c1 ?p1)
            (on ?c2 ?p2)
            (adjacent ?r ?d)
            (between ?h ?d ?r)
        ))
    )
))
(:constraints (and
    (preference bounceBallOffBinToDoggieBed
        (exists (?b - (either dodgeball basketball) ?h - hexagonal_bin ?d - doggie_bed)
            (then 
                (once (agent_holds ?b))
                (hold-while 
                    (and 
                        (in_motion ?d) 
                        (not (agent_holds ?d)) 
                        (not (exists (?o - (either tall_cylindrical_block pyramid_block large_triangular_ramp) ((touch ?o ?d)))))  
                    )
                    (on ?h ?b)
                )     
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        ) 
    )
))
(:scoring maximize (count-nonoverlapping bounceBallOffBinToDoggieBed)
))

; 28 also requires counting events that happen, either as part of the preference or scoring

; so does 29

; 30 appears invalid? I'm not sure I understand it?

;31 is valid and very similar to something we had before

(define (game many-objects-31) (:domain many-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference agentOnRampOnEdge
        (exists (?r - large_triangular_ramp) 
            (and
                (object_orientation ?r edge) 
                (on ?r agent)
            )   
        )
    )
))
; TODO: is this count-total?
(:scoring maximize (+
    (* 10 (> (count-longest agentOnRampOnEdge) 10))
    (* 10 (> (count-longest agentOnRampOnEdge) 20))
    (* 10 (> (count-longest agentOnRampOnEdge) 30))
    (* 10 (> (count-longest agentOnRampOnEdge) 40))
    (* 10 (> (count-longest agentOnRampOnEdge) 50))
    (* 10 (> (count-longest agentOnRampOnEdge) 60))
))
