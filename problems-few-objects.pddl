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


(define (game few-objects-10) (:domain few-objects-room-v1)
(:setup  
; no real setup for 10 unless we want to mark which objects are in the game
)
(:constraints (and 
    (preference chairHitFromBedWithDoggieBed
        (exists (?c - chair ?d - doggie_bed)
            (then 
                (once (and (agent_holds ?d) (on bed agent)))
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

(define (game few-objects-12) (:domain few-objects-room-v1)
(:setup  
; no real setup for 12 since the bin moves
)
(:constraints (and 
    (preference throwToBinOnBed
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (on ?h ?d) (not (in_motion ?d)) (on bed ?h)))
            )
        )
    )
    (preference throwToBinOnDesk
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (on ?h ?d) (not (in_motion ?d)) (on desk ?h)))
            )
        )
    )
    (preference throwToBinOnShelf
        (exists (?d - dodgeball ?h - hexagonal_bin ?s - shelf)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (on ?h ?d) (not (in_motion ?d)) (on ?s ?h)))
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 1 (count-nonoverlapping throwToBinOnBed))
    (* 2 (count-nonoverlapping throwToBinOnDesk))
    (* 3 (count-nonoverlapping throwToBinOnShelf))
)))

(define (game few-objects-13) (:domain few-objects-room-v1)
(:setup  
; no real setup for 13 
)
(:constraints (and 
    (preference onChairFromWallToWall
        (exists (?c - chair ?w1 - wall ?w2 - wall)
            (then
                (once (adjacent agent ?w1))
                (hold (on ?c agent))
                (once (and (adjacent agent ?w2) (opposite ?w1 ?w2)))
            )
        )
    )
)) 
(:scoring minimize (+ 
    (* 1 (count-shortest onChairFromWallToWall))
)))

(define (game few-objects-14) (:domain few-objects-room-v1)
(:setup  
    ; is there a prettier way to do this? 
    (exists (?w1 ?w2 - window ?c - chair ?x1 ?w2 wall  
            ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
        (and
            (not (on floor rug))
            (adjacent bed ?w1)
            (adjacent desk ?w2)
            (adjacent ?c ?x1)
            (opposite ?x1 ?x2)
            (=
                (distance ?x1 ?b1)
                (distance ?b1 ?b2)
                (distance ?b3 ?b4)
                (distance ?b4 ?b5)
                (distance ?b5 ?b6)
                (distance ?b6 ?x2)
            )
        )
    )
)
(:constraints (and 
    (preference onChairFromWallToBlock
        (exists (?c - chair ?w - wall ?b - cube_block)
            (then
                (once (adjacent agent ?w))
                (hold (on ?c agent))
                (once (adjacent agent ?b))
            )
        )
    )
)) 
(:scoring maximize (count-once-per-objects onChairFromWallToBlock)
))


(define (game few-objects-15) (:domain few-objects-room-v1)
(:setup  
; no real setup for 15
)
(:constraints (and 
    (preference throwToWallAndBack
        (exists (?d - dodgeball ?w - wall)
            (then 
                (once (agent_holds ?d))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?d ?w)
                ) 
                (once-measure (agent_holds ?d) (distance agent ?w))
            )
        )
    )
)) 
(:scoring maximize (count-increasing-measure throwToWallAndBack)
))


(define (game few-objects-16) (:domain few-objects-room-v1)
(:setup  
    (forall (?b - cube_block) 
        (> (distance ?b desk) 3)
        (not (exists (?b2 - cube_block) 
            (and 
                (not (= ?b ?b2))
                (< (distance ?b ?b2) 0.5)
            )
        ))
    )
)
(:constraints (and 
    (preference throwHitsBlock
        (exists (?d - dodgeball ?b - cube_block)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (touch ?d ?b))
            )
        )
    )
)) 
(:scoring minimize (+ 
    (* 10 (count-once-per-objects throwHitsBlock))
)))


(define (game few-objects-17) (:domain few-objects-room-v1)
(:setup  
    (exists (?c - curved_wooden_ramp) (adjacent ?c rug)) 

)
(:constraints (and 
    (preference ballLandsOnRed
        (exists (?d - dodgeball ?c - curved_wooden_ramp)
            (then 
                (once (agent_holds ?d) (< (distance agent desktop) 0.5))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                ) 
                (once (and (on rug ?d) (not (in_motion ?d)) (rug_color_under ?d red)))
            )
        )
    )
    (preference blueBallLandsOnPink
        (exists (?d - dodgeball ?c - curved_wooden_ramp)
            (then 
                (once (agent_holds ?d) (< (distance agent desktop) 0.5) (color ?d blue))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                ) 
                (once (and (on rug ?d) (not (in_motion ?d)) (rug_color_under ?d pink)))
            )
        )
    )
    (preference pinkBallLandsOnPink
        (exists (?d - dodgeball ?c - curved_wooden_ramp)
            (then 
                (once (agent_holds ?d) (< (distance agent desktop) 0.5) (color ?d pink))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                ) 
                (once (and (on rug ?d) (not (in_motion ?d)) (rug_color_under ?d pink)))
            )
        )
    )
    (preference ballLandsOnOrangeOrGreen
        (exists (?d - dodgeball ?c - curved_wooden_ramp)
            (then 
                (once (agent_holds ?d) (< (distance agent desktop) 0.5))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                ) 
                (once (and (on rug ?d) (not (in_motion ?d)) (or (rug_color_under ?d green) (rug_color_under ?d orange))))
            )
        )
    )
)) 
(:scoring minimize (+ 
    (* 50 (count-nonoverlapping ballLandsOnRed))
    (* 10 (count-nonoverlapping blueBallLandsOnPink))
    (* 15 (count-nonoverlapping pinkBallLandsOnPink))
    (* 15 (count-nonoverlapping ballLandsOnOrangeOrGreen))
)))



