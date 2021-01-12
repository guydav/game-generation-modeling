; 22 requires a lot of interpretation to figure out what the participant means
; since they refer to "figures" when it's unclear what that would be

; 23 seems impossible but valid, and also requires a fair bit of interpretation

(define (problem setup-23) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (forall (?b - (either bridge_block flat_block)) (on floor ?b)))  
)
)

(define (problem scoring-23) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?g - golfball) (preference bounceBallToMug
        (exists (?m - mug) (exists (?b - (either bridge_block flat_block)) 
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (and (agent_holds ?g) (on bed agent))
                (always-until 
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; the ball touches a block and then lands in/on the mug
                    (sometime-after (touch ?b ?g) (and (on ?m ?g) (not (in_motion ?g))))
                ) 
            )
        )))
    )
))
(:goal (and
    (exists (?m - mug)
        (forall (?g - golfball) 
            (and 
                (thrown ?g) 
                (not (in_motion ?g))
                (on ?m ?g)
            )
        )
    )
))
(:metric maximize (is-violated bounceBallToMug)
))

; TODO: 24 is a juggling game - do we attempt to model it?

(define (problem scoring-24) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    ; TODO; assuming that forall () (preference ... ) attempts to evaluate the preference
    ; once at each time step
    ; TODO: I doubt this actually captures the semantics of juggling -- I think in the 
    ; two ball-case, it's something like both in hand => one in the air => both in the air => 
    ; only second in the air => both in hand
    (forall () (preference twoBallsJuggled
        (exists (?g1 - golfball) (exists (?g2 - golfball) 
            (sometime-after
                ; both balls in hand
                (and (agent_holds ?g1) (agent_holds ?g2))
                (always-until
                    ; first ball is in the air until
                    (and (not (exists (?o - object) (touch ?o ?g1))) (agent_holds ?g2))
                    (always-until
                        ; both balls are in the air 
                        (and (not (exists (?o - object) (touch ?o ?g1))) (not (exists (?o - object) (touch ?o ?g2))) )
                        (always-until
                            ; agent holds first ball while second is in the air
                            (and (agent_holds ?g1) (not (exists (?o - object) (touch ?o ?g2))))
                            ; until both are caught again
                            (and (agent_holds ?g1) (agent_holds ?g2))
                        )
                    )
                )
            )
        ))
    ))
    ; the three ball case is even more complicated -- it's somethhing like:
    ; all three in hand => 1 in air => 1+2 in air => 2 in air => 2+3 in air => 3 in air => all three in hand
    (forall () (preference threeBallsJuggled
        (exists (?g1 - golfball) (exists (?g2 - golfball) (exists (?g3 - golfball)  
            (sometime-after
                ; both balls in hand
                (and (agent_holds ?g1) (agent_holds ?g2) (agent_holds ?g3))
                (always-until
                    ; first ball is in the air while other two are held
                    (and (not (exists (?o - object) (touch ?o ?g1))) (agent_holds ?g2) (agent_holds ?g3))
                    (always-until
                        ; 1+2 in the air
                        (and (not (exists (?o - object) (touch ?o ?g1))) (not (exists (?o - object) (touch ?o ?g2))) (agent_holds ?g3))
                        (always-until
                            ; only 2 in the air
                            (always-until
                                (and (agent_holds ?g1) (not (exists (?o - object) (touch ?o ?g2))) (agent_holds ?g3))
                                (always-until
                                    ; 2 + 3 in the air
                                    (and (agent_holds ?g1) (not (exists (?o - object) (touch ?o ?g2))) (not (exists (?o - object) (touch ?o ?g3))))
                                    ; only 3 in the air
                                    (always-until
                                        (and (agent_holds ?g1) (agent_holds ?g2) (not (exists (?o - object) (touch ?o ?g3))))
                                        ; all 3 in thand
                                        (and (agent_holds ?g1) (agent_holds ?g2) (agent_holds ?g3))
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )))
    ))
))
(:goal (and
    ; all objects were thrown and are now stationary and on something
    (forall (?g - golfball) (exists (?o - object)
        (and 
            (thrown ?g) 
            (not (in_motion ?g))
            (on ?o ?g)
        )
    ))
))
(:metric maximize (+
    ; TODO: this doesn't actually follow proper PDDL, since they don't allow comparisons here
    ; TODO: also, if timesteps are not seconds, this will require rescaling
    (* (10 (/ (is-violated threeBallsJuggled) 30)))
    (* (5 (/ (is-violated twoBallsJuggled) 30)))
    (* (100 (>= (is-violated threeBallsJuggled) 120)))
    (* (50 (>= (is-violated twoBallsJuggled) 120)))
))
)


; TODO: 25 is a balancing game, tricky to model:
(define (problem scoring-25) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    ; TODO; assuming that forall () (preference ... ) attempts to evaluate the preference
    ; once at each time step
    (forall () (preference agentOnRampOnEdge
        (exists (?r - large_triangular_ramp) 
            (and
                (object_orientation ?r edge) 
                (on ?r agent)
            )   
        )
    ))
))
(:goal (and
    ; TODO: this goal state doesn't work -- how do we indicate "agent was on ramp?"
    (forall (?r - large_triangular_ramp) 
        (and 
            (object_orientation ?r face)
            (not (on ?r agent))
        )
    )
))
(:metric maximize (is-violated agentOnRampOnEdge)
))

; 26 is invalid

; TODO: I'm not quite sure how to handle 27 either
; TODO: I could construct a preference mapping onto this entire sequence, but that's ugly
; TODO: I could construct a preference for each part of the circuit, but there's no
; TODO: real way to specify "preference A fulfilled before preference B"
; TODO: also: modeling the "spin in a chair" and "keep beachball in air" actions is nontrivial

(define (problem setup-27) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?r1 - large_triangular_ramp) (exists (?r2 - large_triangular_ramp) 
        (and
            (not (= ?r1 ?r2))
            (<= (distance ?r1 ?r2) 0.5)
        )
    ))
))
)

(define (problem scoring-27) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (preference circuit
        (exists (?r1 - large_triangular_ramp) (exists (?r2 - large_triangular_ramp)
        (exists (?c - chair) (exists (?h - hexagonal_bin) (exists (?b - beachball)
        (forall (?d - dodgeball)
            (sometime-after  
                (sometime-after   
                    ; first, agent starts not between the ramps, then passes between them 
                    ; so after not being between, it is between, then again not between
                    (sometime-after 
                        (not (between ?r1 agent ?r2))
                        (sometime-after (between ?r1 agent ?r2) (not (between ?r1 agent ?r2)))
                    )
                    ; spin four times in a chair
                    (always-until
                        (on ?c agent)
                        ; TODO: there's no clear way to count how many times something happens:
                        (sometime-after
                            (sometime-after
                                (agent_finished_spin)
                                (agent_finished_spin)
                            )
                            (sometime-after
                                (agent_finished_spin)
                                (agent_finished_spin)
                            )
                        )
                    )
                )
                (sometime-after
                    ; throw all dodgeballs into the bin
                    (sometime-after 
                        (agent_holds ?d) 
                        (always-until 
                            (and (not (agent_holds ?d)) (in_motion ?d)) 
                            (and (on ?h ?d) (not (in_motion ?d)))
                        )
                    )
                    ; bounce the beachball for 20 seconds
                    (sometime-after
                        (agent_holds ?b)
                        (always-until
                            ; ball in the air, only touches the agent, which is not a game-object
                            (not (exists (?g - game-object) (or (on ?g ?b) (touch ?g ?b))))
                            ; for at least 20 time-steps, this holds
                            ; TODO: if timesteps are not seconds, this will require rescaling
                            (within 20 (not (exists (?g - game-object) (or (on ?g ?b) (touch ?g ?b)))))
                        )
                    )
                )
            )
        )))))
    )
))
(:goal (and
    ; TODO: is there a clear goal sttate to finishing the circuit?
    ; TODO: I don't think there is one unless we allow is-violated here
    (is-violated circuit)
))
(:metric minimize total-time
))

; Note that 28 is kinda similar to this subject's other game, 18
; TODO: note I could make the setup here more specific by adding additional inferences 

(define (problem setup-28) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?t1 - tall_cylindrical_block) (exists (?t2 - tall_cylindrical_block) 
            (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
        (and
            (not (= ?t1 ?t2))
            (<= (distance ?t1 ?t2) 1)
            (= (distance ?r ?t1) (distance ?r ?t2))
            (adjacent_side ?h front ?r back)
            (= (distance ?h ?t1) (distance ?h ?t2))
            (< (distance ?r ?t1) (distance ?h ?t1))
        )
    ))))
))
)

(define (problem scoring-28) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    ; TODO: what do we 
    (forall (?g - golfball) (preference throwBetweenBlocksToBin
        (exists (?t1 - tall_cylindrical_block) (exists (?t2 - tall_cylindrical_block) 
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
            (sometime-after 
                ; ball starts in hand
                (agent_holds ?g)
                (always-until 
                    ; in motion, not in hand until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; the ball passes between the blocks...
                    (sometime-after (between ?t1 ?g ?t2) 
                        ; and then on the ramp and into the bin
                        (sometime-after (on ?r ?g) (and (on ?h ?g) (not (in_motion ?g))))
                        ; TODO: note that their scoring doesn't actually refer to the ramp
                        ; TODO: only the gameplay does. Should it still be here?
                    )
                ) 
            )
        ) ) ) )
    ))
    (forall (?g - golfball) (preference thrownBallHitBlock
        (exists (?t - tall_cylindrical_block) 
            (sometime-after 
                ; ball starts in hand
                (agent_holds ?g)
                (always-until 
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; the ball touches the block
                    (touch ?g ?t)
                ) 
            )
        ) 
    ))
    (forall (?g - golfball) (preference throwMissesBin
        (exists (?h - hexagonal_bin)
            (sometime-after 
                ; ball starts in hand
                (agent_holds ?g)
                (always-until 
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; ball settles and it's not in/on the bin
                    (and  
                        (not (in_motion ?g))
                        (not (on ?h ?g))
                    )
                ) 
            )
        ) 
    ))
) )
(:goal (and
    (exists (?h - hexagonal_bin)
        (forall (?g - golfball) 
            (and 
                (thrown ?g) 
                (not (in_motion ?g))
                (on ?h ?g)
            )
        )
    )
))
(:metric maximize (+
    (* 5 (is-violated throwBetweenBlocksToBin))
    (- (is-violated thrownBallHitBlock))
    (- (* 2 (is-violated throwMissesBin)))
))

; 29 has no setup

(define (problem scoring-29) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?g - golfball) (preference throwBallToMugThroughRamp
        (exists (?m - mug) (exists (?r - curved_wooden_ramp) 
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?g)
                (always-until 
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; does "slide" mean more than touching it?
                    (sometime-after (touch ?r ?g) (and (on ?h ?g) (not (in_motion ?g))))
                ) 
            )
        )))
    )
    (forall (?g - golfball) (preference throwBallToHexagonalBinThroughRamp
        (exists (?h - hexagonal_bin) (exists (?r - curved_wooden_ramp) 
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?g)
                (always-until 
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; does "slide" mean more than touching it?
                    (sometime-after (touch ?r ?g) (and (on ?h ?g) (not (in_motion ?g))))
                ) 
            )
        )))
    )
))
(:goal (and
    (forall (?g - golfball) 
        (exists (?o - (either mug hexagonal_bin))
            (and 
                (thrown ?g) 
                (not (in_motion ?g))
                (on ?o ?g)
            )
        )
    )
))
(:metric maximize (+
    (* (5 (is-violated throwBallToHexagonalBinThroughRamp)))
    (* (10 (is-violated throwBallToHexagonalBinThroughRamp)))
))
)


; I honestly don't know if I understand 30

; 31 is invalid
