(define (game few-objects-2) (:domain few-objects-room-v1)
(:setup  
    (exists (?h - hexagonal_bin ?c - chair) 
        (game-conserved (< (distance ?h ?c) 1))
    )
    
)
(:constraints (and 
    (preference basketWithChairInTheWay
        (exists (?c - chair ?h - hexagonal_bin ?d - dodgeball) ; (exists (?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d))
                (hold (between agent ?c ?h))
                (once (and (on ?h ?d) (not (agent_holds ?d))))
            )
        ) ;) 
    ) 
    (preference basketMade
        (exists (?h - hexagonal_bin ?d - dodgeball ) ; (exists (?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d))
                (any)
                (once (and (on ?h ?d) (not (agent_holds ?d))))
            )
        )
    ) 
)) 
(:scoring maximize (+ 
    (* 2 (count-nonoverlapping basketWithChairInTheWay))
    (* 1 (count-nonoverlapping chairBetweenAgentAndBall))
))
)

(define (game few-objects-3) (:domain few-objects-room-v1)
(:setup
    (forall (?c - (either desktop laptop)) 
        (game-conserved (not (on desk ?c)))
    )
)
(:constraints (and 
    (preference cubeBlockOnDesk (exists (?c - cube_block) 
        (at-end
            (and 
                (in_building tower ?c)
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on desk ?c)
            )
        )
    ))
    (preference cubeBlockOnCubeBlock (exists (?b - cube_block ?c - cube_block)
        (at-end
            (and 
                (in_building tower ?c)
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on ?b ?c)
            )
        )
    ))) 
))
(:scoring maximize (+ 
    (count-once cubeBlockOnDesk)
    (count-once-per-objects cubeBlockOnCubeBlock)
))
)

(define (game few-objects-4) (:domain few-objects-room-v1)
(:setup (and
    (exists (?w - wall ?h - hexagonal_bin) 
            (game-conserved (= (distance ?w ?h) 1))
    )
))
(:constraints (and 
    (preference throwToWallToBin
        (exists (?d - dodgeball ?w - wall ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?d)) ; ball starts in hand
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?w ?d)
                ) 
                (once  (and (on ?h ?d) (not ((in_motion ?d))))) ; touches wall before in bin
            )
        )
    )
) )
(:scoring maximize (count-nonoverlapping throwToWallToBin)) 
)



(define (game few-objects-5) (:domain few-objects-room-v1)
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin ?d - dodgeball ?t - textbook) 
        (and
            (game-conserved (adjacent_side ?h front ?c back))
            (game-conserved (= (distance_side ?t center ?c front) 1))
            (game-optional (adjacent ?d ?t))
        )
    )
))
(:constraints (and 
    (preference kickBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin ?t - textbook)
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
(:scoring maximize (count-nonoverlapping throwToWallToBin))
)

;6 is invalid


(define (game few-objects-7) (:domain few-objects-room-v1)
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin) 
        (and
            (game-conserved (adjacent_side ?h front ?c back))
            (game-conserved(= (distance_side ?c center room center) 1))
        )
    )
))
(:constraints (and 
    (preference bowlBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin) 
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
(:scoring maximize (* 5 (count-nonoverlapping bowlBallToBin)))
)


(define (game few-objects-8) (:domain few-objects-room-v1)
(:goal (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin) 
        (game-conserved (adjacent_side ?h front ?c back))
    )
))
(:constraints (and 
    (preference rollBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?d)) ; agent starts by holding ball
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...
                    (on ?r ?d) ; on ramp and then in bin -- should this be touch?
                )
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
            )
        ) 
    )
)) 
(:scoring maximize (* 5 (count-nonoverlapping rollBallToBin)))
)

(define (game few-objects-9) (:domain few-objects-room-v1)
(:setup  
; no real setup for 9 unless we want to mark which objects are in the game
)
(:constraints (and 
    (preference cellPhoneThrownOnDoggieBed
        (exists (?d - doggie_bed ?c - cellphone) 
            (then 
                (once (agent_holds ?c))
                (hold (and (not (agent_holds ?c)) (in_motion ?c))) ; in motion, not in hand until...
                (once (and (on ?d ?c) (not (in_motion ?c))))
            )
        )
    )
    (preference textbookThrownOnDoggieBed
        (exists (?d - doggie_bed ?t - textbook) 
            (then 
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t))) ; in motion, not in hand until...
                (once (and (on ?d ?t) (not (in_motion ?t))))
            )
        )
    )
    (preference laptopThrownOnDoggieBed
        (exists (?d - doggie_bed ?l - laptop)
            (then 
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t))) ; in motion, not in hand until...
                (once (and (on ?d ?t) (not (in_motion ?t))))
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 15 (count-nonoverlapping cellPhoneThrownOnDoggieBed))
    (* 10 (count-nonoverlapping textbookThrownOnDoggieBed))
    (* 5 (count-nonoverlapping laptopThrownOnDoggieBed))
)))



(define (problem scoring-10) (:domain few-objects-room-v1)
(:setup  
; no real setup for 10 unless we want to mark which objects are in the game
)
(:constraints (and 
    (preference chairHitFromBedWithDoggieBed
        (exists (?c - chair ?d - doggie_bed)
            (then 
                (once (agent_holds ?d) (on bed agent))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (touch ?d ?c))
            )
        )
    )
    (preference chairHitFromBedWithPillow
        (exists (?c - chair ?p - pillow)
            (then 
                (once (agent_holds ?p) (on bed agent))
                (hold (and (not (agent_holds ?p)) (in_motion ?p))) 
                (once (touch ?p ?c))
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 20 (count-nonoverlapping chairHitFromBedWithDoggieBed))
    (* 20 (count-nonoverlapping chairHitFromBedWithPillow))
)))

; 11 is invalid
