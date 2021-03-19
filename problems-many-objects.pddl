; 3 requires a lot of interpretation to figure out what the participant means
; since they refer to "figures" when it's unclear what that would be

; 4 seems impossible but valid, and also requires a fair bit of interpretation

(define (game many-objects-4) (:domain many-objects-room-v1)
(:setup (and
    (forall (?b - (either bridge_block flat_block)) (game-optional (on floor ?b))))
)
(:constraints (and 
    (preference bounceBallToMug
        (exists (?g - golfball ?m - mug ?b - (either bridge_block flat_block)) 
            (then 
                ; ball starts in hand, with the agent on the chair, near the desk
                (once (and (agent_holds ?g) (on bed agent))
                (hold-while
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; the ball touches a block and then lands in/on the mug
                    (touch ?b ?g)
                ) 
                (once (and (on ?m ?g) (not (in_motion ?g))))
            )
        )
    )
))
(:scoring maximize (count-nonoverlapping bounceBallToMug)
))

; TODO: 5 is a juggling game - do we attempt to model it?

(define (game many-objects-5) (:domain many-objects-room-v1)
(:setup
)
(:constraints (and 
    (forall () (preference twoBallsJuggled
        (exists (?g1 - golfball) (exists (?g2 - golfball) 
            (then
                ; both balls in hand
                ; (and (agent_holds ?g1) (agent_holds ?g2))
                ; first ball is in the air, the second in hand
                (hold (and (not (exists (?o - object) (touch ?o ?g1))) (agent_holds ?g2)))
                ; both balls are in the air 
                (hold (and (not (exists (?o - object) (touch ?o ?g1))) (not (exists (?o - object) (touch ?o ?g2))) ))
                ; agent holds first ball while second is in the air
                (hold (and (agent_holds ?g1) (not (exists (?o - object) (touch ?o ?g2)))))
                ; both are in the air
                (hold (and (not (exists (?o - object) (touch ?o ?g1))) (not (exists (?o - object) (touch ?o ?g2))) ))
            )
        ))
    ))
    ; the three ball case is even more complicated -- it's somethhing like:
    ; all three in hand => 1 in air => 1+2 in air => 2 in air => 2+3 in air => 3 in air => all three in hand
    (forall () (preference threeBallsJuggled
        (exists (?g1 - golfball) (exists (?g2 - golfball) (exists (?g3 - golfball)  
            (then
                ; both balls in hand
                ; (and (agent_holds ?g1) (agent_holds ?g2) (agent_holds ?g3))
                ; first ball is in the air while other two are held (throw the first ball)
                (hold (and (not (exists (?o - object) (touch ?o ?g1))) (agent_holds ?g2) (agent_holds ?g3))) 
                ; 1+2 in the air, 3 in hand (throw the second ball)
                (hold (and (not (exists (?o - object) (touch ?o ?g1))) (not (exists (?o - object) (touch ?o ?g2))) (agent_holds ?g3)))
                ; 2 in air, 1+3 in hand (catch the first ball)
                (hold (and (agent_holds ?g1) (not (exists (?o - object) (touch ?o ?g2))) (agent_holds ?g3)))
                ; 2 + 3 in the air, 1 in hand (throw the third ball)
                (hold (and (agent_holds ?g1) (not (exists (?o - object) (touch ?o ?g2))) (not (exists (?o - object) (touch ?o ?g3)))))
                ; 3 in the air, 1+2 in hand (catch the second ball)
                (hold (and (agent_holds ?g1) (agent_holds ?g2) (not (exists (?o - object) (touch ?o ?g3)))))
                ; 1+3 in the air, 2 in hand (throw the first ball)
                (hold (and (not (exists (?o - object) (touch ?o ?g1))) (agent_holds ?g2) (not (exists (?o - object) (touch ?o ?g3)))))
                ; the next condition in the cycle would be the first one, 1 in the air while 2+3 are in hand (catch the third ball)
            )
        )))
    ))
))
(:metric maximize (+
    ; TODO: maybe this is count-total?
    (* (10 (/ (count-longest threeBallsJuggled) 30)))
    (* (5 (/ (count-longest twoBallsJuggled) 30)))
    (* (100 (>= (count-longest threeBallsJuggled) 120)))
    (* (50 (>= (count-longest twoBallsJuggled) 120)))
))
)


; TODO: 6 is a balancing game, tricky to model:
(define (game many-objects-6) (:domain many-objects-room-v1)
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
(:metric maximize (count-longest agentOnRampOnEdge)
))

; 7 is invalid


(define (game scoring-8) (:domain many-objects-room-v1)
(:setup (and
    (exists (?r1 - large_triangular_ramp ?r2 - large_triangular_ramp) 
        (and
            (not (= ?r1 ?r2))
            (<= (distance ?r1 ?r2) 0.5)
        )
    )
))
(:constraints (and 
    (preference circuit
        (exists (?r1 - large_triangular_ramp ?r2 - large_triangular_ramp ?c - chair ?h - hexagonal_bin ?b - beachball)
            (then 
                ; first, agent starts not between the ramps, then passes between them 
                ; so after not being between, it is between, then again not between
                (not (between ?r1 agent ?r2))
                (any)
                (between ?r1 agent ?r2) 
                (any)
                (not (between ?r1 agent ?r2))
                (any)
                ; spin four times in a chair
                (hold-while
                    (on ?c agent)
                    ; TODO: there's no clear way to count how many times something happens:
                    (agent_finished_spin)
                    (agent_finished_spin)
                    (agent_finished_spin)
                    (agent_finished_spin)
                )
                (any)
                ; throw all dodgeballs into the bin
                (forall-sequence (?d - dodgeball)
                    (then
                        (once (agent_holds ?d))
                        (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                        (once (and (on ?h ?d) (not (in_motion ?d))))
                        (any)  ; to allow for a gap before the next dodgeball is picked up
                    )
                )
                ; bounce the beachball for 20 seconds
                (hold-for 20 (not (exists (?g - game-object) (or (on ?g ?b) (touch ?g ?b)))))
            )
        )
    )
))
(:metric maximize (+
    (* (13 (count-once circuit)))
    (* (2 (<= total-time 60) (count-shortest circuit)))
    (* (3 (<= total-time 50) (count-shortest circuit)))
    (* (2 (<= total-time 40) (count-shortest circuit)))
)
))

(define (game scoring-9) (:domain many-objects-room-v1)
(:setup (and
    (exists (?t1 - tall_cylindrical_block ?t2 - tall_cylindrical_block ?r - curved_wooden_ramp ?h - hexagonal_bin) 
        (and
            (not (= ?t1 ?t2))
            (<= (distance ?t1 ?t2) 1)
            (= (distance ?r ?t1) (distance ?r ?t2))
            (adjacent_side ?h front ?r back)
            (= (distance ?h ?t1) (distance ?h ?t2))
            (< (distance ?r ?t1) (distance ?h ?t1))
        )
    )
))
(:constraints (and 
    (preference throwBetweenBlocksToBin
        (exists (?g - golfball ?t1 - tall_cylindrical_block ?t2 - tall_cylindrical_block ?r - curved_wooden_ramp ?h - hexagonal_bin)
            (then 
                ; ball starts in hand
                (once (agent_holds ?g))
                (hold-while 
                    ; in motion, not in hand until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; the ball passes between the blocks...
                    (between ?t1 ?g ?t2) 
                    ; and then on the ramp 
                    (on ?r ?g)
                )
                ; and into the bin
                (and (on ?h ?g) (not (in_motion ?g)))
            ) 
        )
    ))
    (preference thrownBallHitBlock
        (exists (?g - golfball ?t - tall_cylindrical_block) 
            (then
                ; ball starts in hand
                (once (agent_holds ?g))
                ; in motion, not in hand until...
                (hold (and (not (agent_holds ?g)) (in_motion ?g))) 
                ; the ball touches the block
                (once (touch ?g ?t)) 
            )
        ) 
    ))
    (preference throwMissesBin
        (exists (?g - golfball ?h - hexagonal_bin)
            (then
                ; ball starts in hand
                (once (agent_holds ?g))
                ; ball not in hand and in motion until...
                (hold (and (not (agent_holds ?g)) (in_motion ?g)))
                ; ball settles and it's not in/on the bin
                (once (and (not (in_motion ?g)) (not (on ?h ?g))))
            )
        ) 
    ))
) )
(:metric maximize (+
    (* 5 (count-nonoverlapping throwBetweenBlocksToBin))
    (- (count-nonoverlapping thrownBallHitBlock))
    (- (* 2 (count-nonoverlapping throwMissesBin)))
))

; 10 has no setup

(define (game many-objects-10) (:domain many-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference throwBallToMugThroughRamp
        (exists (?g - golfball ?m - mug ?r - curved_wooden_ramp)  
            (then 
                ; ball starts in hand
                (once (agent_holds ?g))
                ; ball not in hand and in motion until...
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?r ?g)
                )
                (once (and (on ?m ?g) (not (in_motion ?g)))) 
            )
        ))
    ))
    (preference throwBallToHexagonalBinThroughRamp
        (exists (?g - golfball ?h - hexagonal_bin ?r - curved_wooden_ramp) 
            (then 
                ; ball starts in hand
                (once (agent_holds ?g))
                ; ball not in hand and in motion until...
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?r ?g)
                )
                (once (and (on ?h ?g) (not (in_motion ?g)))) 
            )
        ))
    ))
))
(:metric maximize (+
    (* (5 (count-nonoverlapping throwBallToHexagonalBinThroughRamp)))
    (* (10 (count-nonoverlapping throwBallToHexagonalBinThroughRamp)))
))
)

; I honestly don't know if I understand 11

; 12 is invalid
