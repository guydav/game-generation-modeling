(define (problem setup-2) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?h - hexagonal_bin) 
        (exists (?c - chair) 
            (< (distance ?h ?c) 1)
        )
    )
))
;un-comment the following line if metric is needed
;(:metric minimize (???))
)

(define (problem scoring-2) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?d - dodgeball) (preference chairBetweenAgentAndBall
        (exists (?c - chair) (exists (?h - hexagonal_bin)
            ; TODO: change-to-always-until
            (then 
                (once (agent_holds ?d))
                (hold (between agent ?c ?h))
                (once (and (on ?h ?d) (not (agent_holds ?d))))
            )
        ) ) 
    ) )
    (forall (?d - dodgeball) (preference basketMade
        (then 
            (once (agent_holds ?d))
            (any)
            (once (and (on ?h ?d) (not (agent_holds ?d))))
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
(:metric maximize (+ 
    (* 2 (is-violated basketsMade))
    (* 1 (is-violated chairBetweenAgentAndBall))
))
)

(define (problem setup-3) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (forall (?c - (either desktop laptop)) 
        (not (on desk ?c))
    )

))
)

(define (problem scoring-3) (:domain game-v1)
(:objects  ; we'd eventually populate by script
    tower - building
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    ; Here we have the preference before the quantifier, to count it at most once
    (preference cubeBlockOnDesk (exists (?c - cube_block) 
        ; TODO: two options for how to think about this game: 
        ; if we count only at the end of the episode, as done here, then it's a little simpler
        ; if we could also in the middle of the episode, it becomes harder
        (at-end
            (and 
                (in_building tower ?c)
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on desk ?c)
            )
        )
    ))
    ; Here we have the quantifier before, to count how many times it happens 
    (forall (?c - cube_block) (preference cubeBlockOnCubeBlock (exists (?b - cube_block)
        ; TODO: same caveats as above
        (at-end
            (and 
                (in_building tower ?c)
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on ?b ?c)
            )
        )
    ))) 
))
((:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (+ 
    (is-violated cubeBlockOnDesk)
    (is-violated cubeBlockOnCubeBlock)
))
)

(define (problem setup-4) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?w - wall) 
        (exists (?h - hexagonal_bin) 
            (= (distance ?w ?h) 1)
        )
    )
))
)

(define (problem scoring-4) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?d - dodgeball) (preference throwToWallToBin
        (exists (?w - wall) (exists (?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d)) ; ball starts in hand
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?w ?d)
                ) 
                (once  (and (on ?h ?d) (not ((in_motion ?d))))) ; touches wall before in bin
            )
        ) ) 
    ) )
) )
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (is-violated throwToWallToBin)) 
)

(define (problem setup-5) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?c - curved_wooden_ramp) 
        (exists (?h - hexagonal_bin) 
            (exists (?d - dodgeball)
                (exists (?t - textbook)
                    (and
                        (adjacent_side ?h front ?c back)
                        (= (distance_side ?t center ?c front) 1)
                        (adjacent ?d ?t)
                    )
                )
            )
        )
    )
))
)

(define (problem scoring-5) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?d - dodgeball) (preference kickBallToBin
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin) (exists (?t - textbook)
            (then 
                ; agent starts by touching ball while next to the marking textbook
                (once (and (adjacent agent ?t) (touch agent ?d)))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...
                    (on ?r ?d)   ; on ramp and then in bin -- should this be touch?
                ) 
                (once (and (on ?h ?d) (not (in_motion ?d))))
            )
        ) ) 
    )))
))
(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))
(:metric maximize (is-violated throwToWallToBin))
)

;6 is invalid

(define (problem setup-7) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?c - curved_wooden_ramp) 
        (exists (?h - hexagonal_bin) 
            (and
                (adjacent_side ?h front ?c back)
                (= (distance_side ?c center room center) 1)
            )
        )
    )
))

(define (problem scoring-7) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?d - dodgeball) (preference bowlBallToBin
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d)) ; agent starts by holding ball
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...
                    (on ?r ?d)  ; on ramp and then in bin -- should this be touch?
                )
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
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
(:metric maximize (* 5 (is-violated bowlBallToBin)))
)

(define (problem setup-8) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:goal (and
    (exists (?c - curved_wooden_ramp) 
        (exists (?h - hexagonal_bin) 
            (and
                (adjacent_side ?h front ?c back)
            )
        )
    )
))

(define (problem scoring-8) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?d - dodgeball) (preference rollBallToBin
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d)) ; agent starts by holding ball
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...
                    (on ?r ?d) ; on ramp and then in bin -- should this be touch?
                )
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
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
(:metric maximize (* 5 (is-violated rollBallToBin)))
)

; no real setup for 9 unless we want to mark which objects are in the game

(define (problem scoring-9) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (preference cellPhoneThrownOnDoggieBed
        (exists (?d - doggie_bed) (exists (?c - cellphone)
            (then 
                (once (agent_holds ?c))
                (hold (and (not (agent_holds ?c)) (in_motion ?c))) ; in motion, not in hand until...
                (once (and (on ?d ?c) (not (in_motion ?c))))
            )
        ))
    )
    (preference textbookThrownOnDoggieBed
        (exists (?d - doggie_bed) (exists (?t - textbook)
            (then 
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t))) ; in motion, not in hand until...
                (once (and (on ?d ?t) (not (in_motion ?t))))
            )
        ))
    )
    (preference laptopThrownOnDoggieBed
        (exists (?d - doggie_bed) (exists (?l - laptop)
            (then 
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t))) ; in motion, not in hand until...
                (once (and (on ?d ?t) (not (in_motion ?t))))
            )
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
    (* 15 (is-violated cellPhoneThrownOnDoggieBed))
    (* 10 (is-violated textbookThrownOnDoggieBed))
    (* 5 (is-violated laptopThrownOnDoggieBed))
)))

; no real setup for 10 unless we want to mark which objects are in the game

(define (problem scoring-10) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?d - doggie_bed) (preference chairHitFromBedWithDoggieBed
        ; TODO: change-to-always-until
        (exists (?c - chair)
            (then 
                (once (agent_holds ?d) (on bed agent))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (touch ?d ?c))
            )
        )
    ))
    (forall (?p - pillow) (preference chairHitFromBedWithPillow
        ; TODO: change-to-always-until
        (exists (?c - chair)
            (then 
                (once (agent_holds ?p) (on bed agent))
                (hold (and (not (agent_holds ?p)) (in_motion ?p))) 
                (once (touch ?p ?c))
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
)))
))
(:metric maximize (+ 
    (* 20 (is-violated chairHitFromBedWithDoggieBed))
    (* 20 (is-violated chairHitFromBedWithPillow))
)))

; 11 is invalid




