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
            (then 
                ; ball starts in hand, with the agent on the chair, near the desk
                (once (and (agent_holds ?g) (on bed agent))
                (hold-while
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; the ball touches a block and then lands in/on the mug
                    (touch ?b ?g)
                ) 
                (once  (and (on ?m ?g) (not (in_motion ?g))))
            )
        )))
    )
))
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
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
    ; TODO: we'd want to either specify to count all states (with overlap) where this holds
    ; TODO: or to count that this can cycle, that is, the first condition can repeat after the last one
    ; TODO: or perhaps a loop of this condition below in the middle, starting and ending with either 
    ; TODO: both balls in hand or one of the on the ground (or on any other object)
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
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
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
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
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
                (forall (?d - dodgeball)
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
        )))))
    )
))
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (+
    (* (13 (is-violated circuit)))
    (* (2 (<= total-time 60) (is-violated circuit)))
    (* (3 (<= total-time 50) (is-violated circuit)))
    (* (2 (<= total-time 40) (is-violated circuit)))
)
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
    
    (forall (?g - golfball) (preference throwBetweenBlocksToBin
        (exists (?t1 - tall_cylindrical_block) (exists (?t2 - tall_cylindrical_block) 
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
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
        ) ) ) )
    ))
    (forall (?g - golfball) (preference thrownBallHitBlock
        (exists (?t - tall_cylindrical_block) 
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
    (forall (?g - golfball) (preference throwMissesBin
        (exists (?h - hexagonal_bin)
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
)
(:constraints (and 
    (forall (?g - golfball) (preference throwBallToMugThroughRamp
        (exists (?m - mug) (exists (?r - curved_wooden_ramp) 
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
    (forall (?g - golfball) (preference throwBallToHexagonalBinThroughRamp
        (exists (?h - hexagonal_bin) (exists (?r - curved_wooden_ramp) 
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
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (+
    (* (5 (is-violated throwBallToHexagonalBinThroughRamp)))
    (* (10 (is-violated throwBallToHexagonalBinThroughRamp)))
))
)


; I honestly don't know if I understand 30

; 31 is invalid
