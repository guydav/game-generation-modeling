(define (problem setup-12) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
; Interpreting the goal as flat blocks on ground and bridge blocks on flat blocks 
(:goal (and
    (forall (?f - flat_block) (on floor ?f))
    (forall (?b - bridge_block) 
        (exists (?f - flat_block) 
            (on ?f ?b)
        )
    )
))
)

; create tunnels with the brigde blocks and throw the balls
; "1 point for teach"

(define (problem scoring-12) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?b - basketball) (preference throwBallUnderBridge
        (exists (?bb - bridge_block)  
            ; TODO: change-to-always-until ?
            (sometime-after 
                ; ball starts in hand, not under the bridge
                (and (agent_holds ?b) (not (under ?bb ?b)))
                ; Semantically, the block below means that the first condition
                ; holds in all states until we find a pair of states satisfying
                ; the second condition
                (always-until 
                    ; neither ball nor block in hand until...
                    (and (in_motion ?b) (not (agent_holds ?b)) (not (agent_holds ?bb))) 
                    ; the ball is under the bridge and then again not under the bridge
                    (sometime-after (under ?bb ?b) (not (under ?bb ?b))) 
                ) 
            )
        ) 
    ) )
) )
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (is-violated throwBallUnderBridge)) 
)

(define (problem setup-13) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?s - shelf) (exists (?h - hexagonal_bin)
        (and
            (on ?s ?h)
            (forall (?s2 - shelf) (>= (distance ?s desk) (distance ?s2 desk)))
        )
    ))
))
)

(define (problem scoring-13) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?b - basketball) (preference throwBallFromChairToBin
        (exists (?c - chair) (exists (?h - hexagonal_bin) 
            ; TODO: theoretically, we'd have to repeat the setup constraints here too
            ; TODO: to make sure we throw to the same bin, at the same place, right?
            (always-until
                ; ball starts in hand, with the agent on the chair, near the desk
                (always-until 
                    (and (agent_holds ?b) (on ?c agent) (adjacent ?c desk) (agent_perspective upside_down))
                    ; ball not in hand until...
                    (and (not (agent_holds ?b)) (in_motion ?b))
                )
                ; the ball is in the bin
                (and (on ?h ?b) (not (in_motion ?b))) 
            )
        ) ) 
    ))
) )
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (is-violated throwBallFromChairToBin))
)

(define (problem setup-14) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (forall (?b - block) (on floor ?b))
))
)

(define (problem scoring-14) (:domain game-v1)
; TODO: adding this here to show how we'd handle problems with buildings
; an alternative would be to define a few buildings (b1 b2 b3...) in the domain constants
; which doesn't require the scoring modeling to handle instantiating them
(:objects  ; we'd eventually populate by script
    tower - building  
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and
    ; Here we have the preference before the quantifier, to count it at most once
    (preference blockOnFloor (exists (?b - block) 
        (always-until
            (on floor ?b)
            (always-until
                (and
                    (on floor ?b)
                    (in_building tower ?b))
                )
                (building_fell tower)
            )
        )
    )
    ; Here we have the quantifier before, to count how many times it happens 
    (forall (?b - block) (preference blockOnBlock
        (sometime-after
            ; agent starts holding the block
            (agent_holds ?b - block)
            ; at some point after the agent holds the block, the following holds:
            (always-until
                ; we can find a second block such that:
                (exists (?b2 - block)
                    (and 
                        ; both blocks are in the tower
                        (in_building tower ?b)
                        (in_building tower ?b2)
                        ; this new block ?b is on top of the second block ?b2
                        (on ?b ?b2) ; an object cannot be on itself, so this fails if ?b = ?b2
                    )
                )
                ; until the tower falls
                (building_fell tower)
            )
        )      
    ))
    (forall (?b - block) (preference blockFellNear 
        (always-within
            10  ; or whatever is a reasonable time length to resolve the collapse
            (building_fell tower)
            (always-until
                (and (not (agent_holds ?b) (in_motion))) ; block is falling without agent moving it until
                (<= (distance tower ?b) 0.1) ; it settles near the tower
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
(:metric maximize (+
    (is-violated blockOnFloor)
    (is-violated blockOnBlock)
    (- (is-violated blockFellNear))
))
)

; problem 15 has no setup

(define (problem scoring-15) (:domain game-v1)
(:objects  ; we'd eventually populate by script
    ; TODO: adding this here to show how we'd handle problems with buildings
    ; an alternative would be to define a few buildings (b1 b2 b3...) in the domain constants
    tower - building  
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and
    ; Count how many objects are part of the tower
    (forall (?o - game_object) (preference objectInTower 
        (sometime-after 
            (agent_holds ?b)
            (always-until
                (in_building tower ?o)
                (building_fell tower)
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
(:metric maximize (/ (* 100 (max_height tower)) (is-violated objectInTower))
)
)

;16 is invalid

(define (problem setup-17) (:domain game-v1)
(:objects  ; we'd eventually populate by script
    
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?h - hexagonal_bin) (exists (?d - doggie_bed) (exists (?p - pillow)
        and(
            (<= (distance ?h ?d) 1)
            (<= (distance ?h ?p) 1)
            (<= (distance ?p ?d) 1)
            (< (distance agent ?h) (distance agent ?d))
            (< (distance agent ?h) (distance agent ?p))
            (< (distance agent ?d) (distance agent ?p))
        )
    )))
))
)

(define (problem scoring-17) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and
    (preference beachballToHexagonalBin
        (exists (?b - beachball) (exists (?h - hexagonal_bin)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?h ?b) (not (in_motion ?b)))))
        ))
    )
    (preference beachballToDoggieBed
        (exists (?b - beachball) (exists (?d - doggie_bed)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?d ?b) (not (in_motion ?b)))))
        ))
    )
    (preference beachballToPillow
        (exists (?b - beachball) (exists (?p - pillow)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?p ?b) (not (in_motion ?b)))))
        ))
    )
    (preference dodgeballToHexagonalBin
        (exists (?b - dodgeball) (exists (?h - hexagonal_bin)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?h ?b) (not (in_motion ?b)))))
        ))
    )
    (preference dodgeballToDoggieBed
        (exists (?b - dodgeball) (exists (?d - doggie_bed)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?d ?b) (not (in_motion ?b)))))
        ))
    )
    (preference dodgeballToPillow
        (exists (?b - dodgeball) (exists (?p - pillow)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?p ?b) (not (in_motion ?b)))))
        ))
    )
    (preference basketballToHexagonalBin
        (exists (?b - basketball) (exists (?h - hexagonal_bin)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?h ?b) (not (in_motion ?b)))))
        ))
    )
    (preference basketballToDoggieBed
        (exists (?b - basketball) (exists (?d - doggie_bed)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?d ?b) (not (in_motion ?b)))))
        ))
    )
    (preference basketballToPillow
        (exists (?b - basketball) (exists (?p - pillow)
            (sometime-after (agent_holds ?b) (always-until (and (not (agent_holds ?b)) (in_motion ?b)) (and (on ?p ?b) (not (in_motion ?b)))))
        ))
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
    (* 3 (is-violated beachballToHexagonalBin))
    (* 5 (is-violated beachballToDoggieBed))
    (* 7 (is-violated beachballToPillow))
    (* 6 (is-violated dodgeballToHexagonalBin))
    (* 8 (is-violated dodgeballToDoggieBed))
    (* 10 (is-violated dodgeballToPillow))
    (* 9 (is-violated basketballToHexagonalBin))
    (* 11 (is-violated basketballToDoggieBed))
    (* 13 (is-violated basketballToPillow))
))
)

(define (problem setup-18) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?t1 - tall_cylindrical_block) (exists (?t2 - tall_cylindrical_block) (exists (?tb - teddy_bear)
        (and
            (not (= ?t1 ?t2))
            (<= (distance ?t1 ?t2) 2)
            (= (distance ?tb ?t1) (distance ?tb ?t2))
        )
    )))
))
)

(define (problem scoring-18) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    ; TODO: is the subject refers to it first as throwing, and then as rolling, should we consider it?
    (forall (?b - basketball) (preference throwBetweenBlocksToBear
        (exists (?t1 - tall_cylindrical_block) (exists (?t2 - tall_cylindrical_block) (exists (?tb - teddy_bear)
            (sometime-after 
                (agent_holds ?b)
                (always-until 
                    ; ball not in hand until...
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    ; the ball passes between the blocks and then touches the bear
                    (sometime-after (between ?t1 ?b ?t2) (touch ?b ?tb))
                ) 
            )
        ) ) )
    ))
    (forall (?b - basketball) (preference thrownBallHitBlock
        (exists (?t - tall_cylindrical_block) 
            (sometime-after 
                (agent_holds ?b)
                (always-until 
                    ; ball not in hand until...
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    ; the ball touches the block
                    (touch ?b ?t)
                ) 
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
(:metric maximize (+
    (* 15 (is-violated throwBetweenBlocksToBear))
    (* (- 5) (is-violated thrownBallHitBlock))
))

; 19 has no setup

(define (problem scoring-19) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?o - (either pillow beachball dodgeball)) (preference thrownObjectKnocksDesktop
        (exists (?d - desktop)
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?o)
                (always-until 
                    ; ball not in hand and in until...
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?d)))
                    ; the ball knocks off the desktop
                    (sometime-after (touch ?o ?d) (and (not (on desk ?d)) (not (in_motion ?d))))
                ) 
            )
        ) 
    ))
    (forall (?o - (either pillow beachball dodgeball)) (preference thrownObjectKnocksDeskLamp
        (exists (?d - desk_lamp)
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?o)
                (always-until 
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?d)))
                    ; the ball knocks off the lamp
                    (sometime-after (touch ?o ?d) (and (not (on desk ?d)) (not (in_motion ?d))))
                ) 
            )
        ) 
    ))
    (forall (?o - (either pillow beachball dodgeball)) (preference thrownObjectKnocksCD
        (exists (?c - cd)
            (sometime-after 
                ; ball starts in hand, with the agent on the chair, near the desk
                (agent_holds ?o)
                (always-until 
                    ; ball not in hand until...
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?c)))
                    ; the ball knocks off the CD
                    (sometime-after (touch ?o ?c) (and (not (on desk ?c)) (not (in_motion ?c))))
                ) 
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
(:metric maximize (+
    (* 5 (is-violated thrownObjectKnocksDesktop))
    (* 10 (is-violated thrownObjectKnocksDeskLamp))
    (* 15 (is-violated thrownObjectKnocksCD))
))

; 20 has no setup

(define (problem scoring-20) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?b - basketball) (preference throwBallWithEyesClosed
        (exists (?h - hexagonal_bin) 
            (always-until 
                ; ball starts in hand, with the agent on the chair, near the desk
                (and (agent_holds ?b) (agent_perspective eyes_closed))
                (always-until 
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    ; the ball is in the bin
                    (and (on ?h ?b) (not (in_motion ?b)))
                ) 
            )
        ) 
    ))
) )
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (* 5(is-violated throwBallFromChairToBin)))
)

; TODO: 21 has no setup, and is a little nonsensical, but could be modeled like this:

; TODO: adding this here to show how we'd handle problems with buildings
    ; an alternative would be to define a few buildings (b1 b2 b3...) in the domain constants
(define (problem scoring-21) (:domain game-v1)
(:objects  ; we'd eventually populate by script
    castle - building  
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and
    ; Here we have the preference before the quantifier, to count it at most once
    (forall (?b - block) (preference correctColorBlock 
        (and
            (sometime-after
                (agent_holds ?b)
                (always (in_building castle ?b))
            )
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
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (* (2 (is-violated correctColorBlock)))
)
)


