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
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?g - golfball) (preference bounceBallToMug
        (exists (?m - mug) (exists (?b - (either bridge_block flat_block)) 
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (and (agent_holds ?g) (on bed agent))
                (always-until 
                    ; ball not in hand until...
                    (not (agent_holds ?g))
                    ; the ball touches a block and then lands in/on the mug
                    (sometime-after (touch ?b ?g) (on ?m ?g))
                ) 
            )
        )))
    )
))
(:goal (and
    (forall (?g - golfball) 
        (and 
            (thrown ?g) 
            (not (in_motion ?g))
        )
    )
))
(:metric maximize (is-violated bounceBallToMug)
))

; TODO: 24 is a juggling game - do we attempt to model it?
; TODO: similarly, 25 is a balancing game, do we attempt to model it?
; an attempt to model 25 would look something like

(define (problem scoring-25) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
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
    (episode_over)
    (forall (?r - large_triangular_ramp) (object_orientation ?r face))
))
(:metric maximize (is-violated agentOnRampOnEdge)
))

; 26 is invalid

; TODO: I'm not quite sure how to handle 27 either
; TODO: I could construct a preference mapping onto this entire sequence, but that's ugly
; TODO: I could construct a preference for each part of the circuit, but there's no
; TODO: real way to specify "preference A fulfilled before preference B"
; TODO: also: modeling the "spin in a chair" and "keep beachball in air" actions is nontrivial

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
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    ; TODO: what do we 
    (forall (?g - golfball) (preference throwBetweenBlocksToBin
        (exists (?t1 - tall_cylindrical_block) (exists (?t2 - tall_cylindrical_block) 
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?g)
                (always-until 
                    ; ball not in hand until...
                    (not (agent_holds ?g))
                    ; the ball passes between the blocks...
                    (sometime-after (between ?t1 ?g ?t2) 
                        ; and then on the ramp and into the bin
                        ; TODO: note that their scoring doesn't actually refer to the ramp
                        ; TODO: only the gameplay does. Should it still be here?
                        (sometime-after (on ?r ?g) (on ?h ?g))
                    )
                ) 
            )
        ) ) ) )
    ))
    (forall (?g - golfball) (preference thrownBallHitBlock
        (exists (?t - tall_cylindrical_block) 
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?g)
                (always-until 
                    ; ball not in hand until...
                    (not (agent_holds ?g))
                    ; the ball touches the block
                    (touch ?g ?t)
                ) 
            )
        ) 
    ))
    (forall (?g - golfball) (preference throwMissesBin
        (exists (?h - hexagonal_bin)
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?g)
                (always-until 
                    ; ball not in hand until...
                    (not (agent_holds ?g))
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
    (forall (?g - golfball) 
        (and 
            (thrown ?g) 
            (not (in_motion ?g))
        )
    )
))
(:metric maximize (+
    (* 5 (is-violated throwBetweenBlocksToBin))
    (- (is-violated thrownBallHitBlock))
    (* 2 (- (is-violated throwMissesBin)))
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
                    ; ball not in hand until...
                    (not (agent_holds ?g))
                    ; does "slide" mean more than touching it?
                    (sometime-after (touch ?r ?g) (on ?m ?g))
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
                    ; ball not in hand until...
                    (not (agent_holds ?g))
                    ; does "slide" mean more than touching it?
                    (sometime-after (touch ?r ?g) (on ?h ?g))
                ) 
            )
        )))
    )
))
(:goal (and
    ((forall (?g - golfball) 
        (and 
            (thrown ?g) 
            (not (in_motion ?g))
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
