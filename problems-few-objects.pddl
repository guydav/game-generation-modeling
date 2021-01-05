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
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?d - dodgeball) (preference chairBetweenAgentAndBall
        (exists (?c - chair) (exists (?h - hexagonal_bin)
            (sometime-after (agent_holds ?d) 
                (always-until (between agent ?c ?h) (and (on ?h ?d) (not (agent_holds ?d)))
                )
            )
        ) ) 
    ) )
    (forall (?d - dodgeball) (preference basketMade
        (exists (?h - hexagonal_bin) (sometime (on ?h ?d)))
    ) )
) )
(:goal (and  ; is this the correct goal state? Or should we consider the goal state a time out?
    (episode_over)
    (forall (?d - dodgeball) 
        (and
            (thrown ?d)
            (not (in_motion ?d))
        )
    )
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
    (forall (?d - desktop) 
        (not (on desk ?d))
    )

))
;un-comment the following line if metric is needed
;(:metric minimize (???))
)

(define (problem scoring-3) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    ; Here we have the preference before the quantifier, to count it at most once
    (preference cubeBlockOnDesk (exists (?c - cube_block) 
            (and 
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on desk ?c)
            )
    ))
    ; Here we have the quantifier before, to count how many times it happens 
    (forall (?c - cube_block) (preference cubeBlockOnCubeBlock (exists (?b - cube_block)
            (and 
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on ?b ?c) ; an object cannot be on itself, so this fails if ?b = ?c
            )
    ))) 
))
(:goal (and  ; Game ends either after a timeout or when all blocks are stacked
    (exists (?b - cube_block)
        (and
            (or (object_orientation ?b edge) (object_orientation ?b point))
            (on desk ?b)
            (forall (?c - cube_block)
                (or
                    ; either it's the base cube block
                    (= ?c ?b)  
                    ; or it's on the side and top of another cube_block
                    (and
                        (or (object_orientation ?c edge) (object_orientation ?c point))
                        (exists (?e - cube_block) (on ?e ?c))
                    )
                    
                )
            )
        )
    )
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
            (< (distance ?w ?h) 1)
        )
    )
))

(define (problem scoring-4) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?d - dodgeball) (preference throwToWallToBin
        (exists (?w - wall) (exists (?h - hexagonal_bin)
            (sometime-after (agent_holds ?d) ; ball starts in hand
                (always-until 
                    (not (agent_holds ?d)) ; not in hand until...
                    (sometime-after (touch ?w ?d) (on ?h ?d)) ; touches wall before in bin
                )
            )
        ) ) 
    ) )
) )
(:goal (and  ; is this the correct goal state? Or should we consider the goal state a time out?
    (episode_over)
    (forall (?d - dodgeball) 
        (and
            (thrown ?d)
            (not (in_motion ?d))
        )
    )
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
                        (= (distance_side ?d center ?c front) 1)
                        (adjacent ?d ?t)
                    )
                )
            )
        )
    )
)))

(define (problem scoring-5) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?d - dodgeball) (preference kickBallToBin
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
            (sometime-after 
                (touch agent ?d) ; agent starts by touching ball
                (always-until 
                    (not (agent_holds ?d))  ; not in hand until...
                    (sometime-after (on ?r ?d) (on ?h ?d)) ; on ramp and then in bin -- should this be touch?
                ) 
            )
        ) ) 
    )))
) 
(:goal (and  ; is this the correct goal state? Or should we consider the goal state a time out?
    (episode_over)
    (forall (?d - dodgeball) 
        (and
            (thrown ?d)
            (not (in_motion ?d))
        )
    )
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
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?d - dodgeball) (preference bowlBallToBin
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
            (sometime-after (agent_holds ?d) ; agent starts by holding ball
                (always-until 
                    (not (agent_holds ?d)) ; not in hand until...
                    (sometime-after (on ?r ?d) (on ?h ?d)) ; on ramp and then in bin -- should this be touch?
                ) 
            )
        )) 
    )) 
))
(:goal (and 
    (episode_over)
    (forall (?d - dodgeball) 
        (and
            (thrown ?d)
            (not (in_motion ?d))
        )
    )
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
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?d - dodgeball) (preference rollBallToBin
        (exists (?r - curved_wooden_ramp) (exists (?h - hexagonal_bin)
            (sometime-after (agent_holds ?d) ; agent starts by holding ball
                (always-until 
                    (not (agent_holds ?d)) ; not in hand until in bin
                    (sometime-after (on ?r ?d) (on ?h ?d)) ; on ramp and then in bin -- should this be touch?
                ) 
            )
        )) 
    ))
)) 
(:goal (and  ; TODO: is this the correct goal state? Or should we consider the goal state a time out?
    (episode_over)
    (forall (?d - dodgeball) 
        (and
            (thrown ?d)
            (not (in_motion ?d))
        )
    )
))
(:metric maximize (* 5 (is-violated rollBallToBin)))
)

; no real setup for 9 unless we want to mark which objects are in the game

(define (problem scoring-9) (:domain game-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (preference cellPhoneThrownOnDoggieBed
        (exists (?d - doggie_bed) (exists (?c - cellphone)
            (sometime-after (agent_holds ?c) (always-until (not (agent_holds ?c)) (on ?d ?c)))
        ))
    )
    (preference textbookThrownOnDoggieBed
        (exists (?d - doggie_bed) (exists (?t - textbook)
            (sometime-after (agent_holds ?t) (always-until (not (agent_holds ?t)) (on ?d ?t)))
        ))
    )
    (preference laptopThrownOnDoggieBed
        (exists (?d - doggie_bed) (exists (?l - laptop)
            (sometime-after (agent_holds ?l) (always-until (not (agent_holds ?l)) (on ?d ?l)))
        ))
    )
)) 
(:goal (and  ; TODO: should this be for all of them? or at least one of them?
    (episode_over)
    (forall (?o - (either cellphone textbook laptop))
        (and 
            (thrown ?o)
            (not (in_motion ?o))
        )
    )
)
)
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
    (at 100 (episode_over))  ; assuming that 100 is some reasonable episode length
)
(:constraints (and 
    (forall (?d - doggie_bed) (preference chairHitFromBedWithDoggieBed
        (exists (?b - bed) (exists (?c - chair)
            (sometime-after 
                (and (agent_holds ?d) (on ?b agent))
                (always-until (not (agent_holds ?d)) (touch ?d ?c))
            )
        ))
    ))
    (forall (?p - pillow) (preference chairHitFromBedWithPillow
        (exists (?b - bed) (exists (?c - chair)
            (sometime-after 
                (and (agent_holds ?p) (on ?b agent))
                (always-until (not (agent_holds ?p)) (touch ?p ?c))
            )
        ))
    ))
)) 
(:goal (and  
    (episode_over)
    (forall (?o - (either doggie_bed pillow))
        (and
            (thrown ?o)
            (not (in_motion ?o))
        )
    )
))
(:metric maximize (+ 
    (* 20 (is-violated chairHitFromBedWithDoggieBed))
    (* 20 (is-violated chairHitFromBedWithPillow))
)))

; 11 is invalid




