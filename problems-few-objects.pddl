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
(:scoring maximize (+ 
    (* 10 (count-once-per-objects throwHitsBlock))
)))


(define (game few-objects-17) (:domain few-objects-room-v1)
(:setup  
    (exists (?c - curved_wooden_ramp) (game-conserved (adjacent ?c rug)))

)
(:constraints (and 
    (preference ballLandsOnRed
        (exists (?d - dodgeball ?c - curved_wooden_ramp)
            (then 
                (once (and (agent_holds ?d) (< (distance agent desktop) 0.5)))
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
                (once (and (agent_holds ?d) (< (distance agent desktop) 0.5) (color ?d blue)))
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
                (once (and (agent_holds ?d) (< (distance agent desktop) 0.5) (color ?d pink)))
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
                (once (and (agent_holds ?d) (< (distance agent desktop) 0.5)))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                ) 
                (once (and (on rug ?d) (not (in_motion ?d)) (or (rug_color_under ?d green) (rug_color_under ?d orange))))
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 50 (count-nonoverlapping ballLandsOnRed))
    (* 10 (count-nonoverlapping blueBallLandsOnPink))
    (* 15 (count-nonoverlapping pinkBallLandsOnPink))
    (* 15 (count-nonoverlapping ballLandsOnOrangeOrGreen))
)))

; 18 is a little underspecified -- what do we want to do about it?
; 18 also requires an actual end state -- how do we want to handle that?


(define (game few-objects-19) (:domain few-objects-room-v1)
(:setup  
    (exists (?h - hexagonal_bin ?l - lamp) (game-conserved (on ?h ?l)))
)
(:constraints (and 
    (preference ballHitsLamp
        (exists (?d - dodgeball ?l - lamp ?h - hexagonal_bin)
            (then 
                (once (and (agent_holds ?d) (> (distance agent ?h) 10)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?l) (touch ?d ?l) (not (in_motion ?d))))
            )
        )
    )
)) 
(:scoring maximize (+ (* 10 (count-nonoverlapping ballHitsLamp))
)))

(define (game few-objects-20) (:domain few-objects-room-v1)
(:setup  
    ; is there a prettier way to do this? 
    (and
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
            (game-optional (and
                (on bed ?b1)
                (on bed ?b2)
                (adjacent ?b1 ?b2)
                (or 
                    (on ?b1 ?b3)
                    (on ?b2 ?b3)
                )
                (on bed ?b4)
                (on bed ?b5)
                (adjacent ?b4 ?b5)
                (or 
                    (on ?b4 ?b6)
                    (on ?b5 ?b6)
                )
            ))
        )
        (forall (?d - dodgeball) (on desk ?d))
    )
)
(:constraints (and 
    (preference bothBallsThrownFromDesk
        (then
            (forall-sequence (?d - dodgeball)
                (once (and (agent_holds ?d) (adjancet agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (adjacent agent desk)))
                (once (not (in_motion ?d)))
                (hold (adjacent agent dek))
            )
            (once (forall (?b - cube_block) (not (in_motion ?b))))
        )
    )
    (preference bothYellowBlocksHit
        ; This preference was hard to define because I wanted to allow for all orderings
        ; It could be that the same ball hits both at the same time, or the sam ball hits
        ; both one after the other, or different balls hit both
        (forall (?b - yellow_cube_block)
            (exists (?d - dodgeball)
                (then
                    (hold-while
                        (not (touch agent ?b))
                        (in_motion ?d)
                        (touch ?b ?d)
                        (in_motion ?b)
                        (not (in_motion ?b))
                    )
                )
            )
        )
    )
    (preference bothBlueBlocksHit
        ; see notes on first one
        (forall (?b - blue_cube_block)
            (exists (?d - dodgeball)
                (then
                    (hold-while
                        (not (touch agent ?b))
                        (in_motion ?d)
                        (touch ?b ?d)
                        (in_motion ?b)
                        (not (in_motion ?b))
                    )
                )
            )
        )
    )
    (preference bothBrownBlocksHit
        ; see notes on above
        (forall (?b - brown_cube_block)
            (exists (?d - dodgeball)
                (then
                    (hold-while
                        (not (touch agent ?b))
                        (in_motion ?d)
                        (touch ?b ?d)
                        (in_motion ?b)
                        (not (in_motion ?b))
                    )
                )
            )
        )
    )
    (preference allBlocksHit
        ; see notes on above
        (forall (?b - cube_block)
            (exists (?d - dodgeball)
                (then
                    (hold-while
                        (not (touch agent ?b))
                        (in_motion ?d)
                        (touch ?b ?d)
                        (in_motion ?b)
                        (not (in_motion ?b))
                    )
                )
            )
        )
    )
    (preference allBlocksHitWithSameBall
        ; I don't know if this is legit -- but I think I would specifcy it like this:
        (exists (?d - dodgeball)
            (forall (?b - cube_block)
                (then
                    (hold-while
                        (not (touch agent ?b))
                        (in_motion ?d)
                        (touch ?b ?d)
                        (in_motion ?b)
                        (not (in_motion ?b))
                    )
                )
            )
        )
    )
)) 
; TODO: the end-state here is probably that bothBallsThrownFromDesk or allBlocksHitWithSameBall is satisfied
(:scoring maximize (+ 
    (* 2 (count-once bothYellowBlocksHit))
    (* 2 (count-once bothBlueBlocksHit))
    (* 2 (count-once bothBrownBlocksHit))
    (* 1 (count-once allBlocksHit))
    (* 3 (count-once allBlocksHitWithSameBall))
)))


(define (game few-objects-21) (:domain few-objects-room-v1)
(:setup  
    ; is there a prettier way to do this? 
    (and
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
            (game-optional (and
                (on floor ?b1)
                (on ?b1 ?b2)
                (on floor ?b3)
                (on ?b3 ?b4)
                (on floor ?b5)
                (on ?b5 ?b6)
                (= (distance ?b1 ?b3) (distance ?b1 ?b5))
                (= (distance ?b1 ?b3) (distance ?b3 ?b5))
            ))
        )
    )
)
(:constraints (and 
    (preference blockKnockedFromBlock 
        (exists (?d - dodgeball ?b1 ?b2 - cube_block ?c - chair)
            (then
                (once (and (agent_holds ?d) (on ?c agent) (on floor ?b1) (on ?b1 ?b2)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (touch agent ?b1)) (not (touch agent ?b2)))) 
                (once (and (on floor ?b1) (on floor ?b2)))            
            )
        )
    )
)) 
(:scoring maximize (count-once-per-objects blockKnockedFromBlock))
)

; 22 is invalid

; 23 is a little inconsistent, but should work
(define (game few-objects-23) (:domain few-objects-room-v1)
(:setup  
    (and
        (forall (?b - cube_block) (exists (?b1 ?b2 - cube_block) 
            (game-conserved (and
                (on floor ?b)
                (not (= ?b ?b1))
                (not (= ?b ?b2))
                (not (= ?b1 ?b2))
                (adjacent ?b ?b1)
                (adjacent ?b ?b2)
            ))
        )
    )
)
(:constraints (and 
    (preference pillowLandsInBlocks 
        (exists (?p - pillow)
            (then
                (once (and (agent_holds ?p) (forall (?b - cube_block) (> (distance agent ?b) 2))))
                (hold (and (not (agent_holds ?p)) (in_motion ?p))) 
                (hold-to-end (exists (?b1 ?b2 - cube_block) 
                    (and (on floor ?b1) (on floor ?b2) (on floor ?p) (between ?b1 ?p ?b2 ))
                ))          
            )
        )
    )
    (preference cdLandsInBlocks 
        (exists (?c - cd)
            (then
                (once (and (agent_holds ?c) (forall (?b - cube_block) (> (distance agent ?b) 2))))
                (hold (and (not (agent_holds ?c)) (in_motion ?c))) 
                (hold-to-end (exists (?b1 ?b2 - cube_block) 
                    (and (on floor ?b1) (on floor ?b2) (on floor ?c) (between ?b1 ?c ?b2 ))
                ))          
            )
        )
    )
    (preference dodgeballLandsInBlocks 
        (exists (?d - dodgeball)
            (then
                (once (and (agent_holds ?d) (forall (?b - cube_block) (> (distance agent ?b) 2))))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (hold-to-end (exists (?b1 ?b2 - cube_block) 
                    (and (on floor ?b1) (on floor ?b2) (on floor ?d) (between ?b1 ?d ?b2 ))
                ))
            )
        )
    )
    (preference objectBouncesOut
        (exists (?p - (either pillow cd dodgeball))
            (then
                (once (and (agent_holds ?p) (forall (?b - cube_block) (> (distance agent ?b) 2))))
                (hold-while 
                    (and (not (agent_holds ?p)) (in_motion ?p))
                    (exists (?b1 ?b2 - cube_block) (and (on floor ?b1) (on floor ?b2) (touch floor ?p) (between ?b1 ?p ?b2 )))
                    (and (touch floor ?p) (not (exists (?b1 ?b2 - cube_block) (and (on floor ?b1) (on floor ?b2) (between ?b1 ?p ?b2 )))))    
                )           
            )
        )
    )
)) 
(:scoring maximize (+
    (* 5 (count-once-per-objects pillowLandsInBlocks))
    (* 10 (count-once-per-objects cdLandsInBlocks))
    (* 20 (count-once-per-objects dodgeballLandsInBlocks))
    (* -5 (count-once-per-objects objectBouncesOut))
)))


(define (game few-objects-24) (:domain few-objects-room-v1)
(:setup  
    (forall (?b - cube_block) 
        (not (exists (?b2 - cube_block) 
            (and 
                (not (= ?b ?b2))
                (< (distance ?b ?b2) 1)
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
    (preference throwHitsBlockAndFlips
        (exists (?d - dodgeball ?b - cube_block)
            (then 
                (once (agent_holds ?d))
                (hold-while 
                    (not (agent_holds ?d) (not (agent_holds ?b)))
                    (once (touch ?d ?b))
                    (once (not (object_orientation ?b face)))
                )
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 5 (count-once-per-objects throwHitsBlock))
    (* 10 (count-once-per-objects throwHitsBlockAndFlips))
    ; Without accounting for the "if the cube blocks moves a long distance" bit
)))


; this ignore the "spin X times" bit
(define (game few-objects-25) (:domain few-objects-room-v1)
(:setup  
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block ?c - chair ?h - hexagonal_bin) 
            (game-optional (and 
                (adjacent ?b1 ?b2)
                (adjacent ?b1 ?c)
                (adjacent ?b2 ?c)
                (adjacent ?b3 ?b4)
                (adjacent ?b3 ?h)
                (adjacent ?b4 ?h)
                (adjacent ?b5 ?b6)
                (on ?b5 bed)
                (on ?b6 bed)
            ))
        ))
    )
)
(:constraints (and 
    (preference blocksFromChairToShelf
        (exists (?c - chair ?b1 ?b2 - cube_block ?s - shelf)
            (then 
                (once (adjcent agent ?s))
                (hold (and (agent_perspective eyes_closed) (adjacent ?b1 ?c) (adjacent ?b2 ?c)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1) (agent_holds ?b2)))
                (hold (and (agent_perspective eyes_closed) 
                    (or 
                        (and (agent_holds ?b1) (on ?s ?b2) )
                        (and (on ?s ?b1) (agent_holds ?b2))
                    )
                ))
                (once (and (on ?s ?b1) (on ?s ?b2)))
            )
        )
    )
    (preference blocksFromBinToShelf
        (exists (?h - hexagonal_bin ?b1 ?b2 - cube_block ?s - shelf)
            (then 
                (once (adjcent agent ?s))
                (hold (and (agent_perspective eyes_closed) (adjacent ?b1 ?h) (adjacent ?b2 ?h)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1) (agent_holds ?b2)))
                (hold (and (agent_perspective eyes_closed) 
                    (or 
                        (and (agent_holds ?b1) (on ?s ?b2) )
                        (and (on ?s ?b1) (agent_holds ?b2))
                    )
                ))
                (once (and (on ?s ?b1) (on ?s ?b2)))
            )
        )
    )
    (preference blocksFromBedToShelf
        (exists (?b1 ?b2 - cube_block ?s - shelf)
            (then 
                (once (adjcent agent ?s))
                (hold (and (agent_perspective eyes_closed) (on bed ?b1) (on bed ?b2)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1) (agent_holds ?b2)))
                (hold (and (agent_perspective eyes_closed) 
                    (or 
                        (and (agent_holds ?b1) (on ?s ?b2) )
                        (and (on ?s ?b1) (agent_holds ?b2))
                    )
                ))
                (once (and (on ?s ?b1) (on ?s ?b2)))
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 20 (count-once blocksFromChairToShelf))
    (* 10 (count-once blocksFromBinToShelf))
    (* 5 (count-once blocksFromBedToShelf))
)))


(define (game few-objects-26) (:domain few-objects-room-v1)
(:setup  
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block ?c - chair ?h - hexagonal_bin) 
        (and 
            (game-conserved (and 
                (on bed ?h)
                (adjacent ?c desk)
            ))
            (game-optional (and 
                (adjacent ?b1 bed)
                (on ?b1 ?b2)
                (adjacent ?b3 ?b1)
                (on ?b3 ?b4)
                (adjacent ?b5 ?b3)
                (on ?b5 ?b6)
                (between ?b1 ?b3 ?b5)
            ))
        )
    )
)
(:constraints (and 
    (preference throwHitsBlock
        (exists (?d - dodgeball ?b - cube_block ?c - chair)
            (then 
                (once (and (on ?c agent) (agent_holds ?d)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (hold (touch ?d ?b))
                (once (in_motion ?b))
            )
        )
    )
    (preference throwInBin
        (exists (?d - dodgeball ?h - hexagonal_bin ?c - chair)
            (then 
                (once (and (on ?c agent) (agent_holds ?d)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (on ?h ?d) (not (in_motion ?d))))
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 1 (count-once-per-objects throwHitsBlock))
    (* 5 (count-once-per-objects throwInBin))
)))

(define (game few-objects-27) (:domain few-objects-room-v1)
(:setup  

)
(:constraints (and 
    ; Two valid ways of writing this -- one where I define all of the requirements
    ; in each preference, and another where I define it in the scoring.
    ; I'll actually try doing it in the scoring here, we'll see how well that works
    (preference bookOnChairs
        (exists (?c1 ?c2 - chair ?t - textbook)
            (at-end 
                (and 
                    (on ?c1 ?t) 
                    (on ?c2 ?t)
                )
            )
        )
    )
    (preference firstLayerOfBlocks
        (exists (?t - textbook ?b1 ?b2 ?b3 - cube_block)
            (at-end 
                (and 
                    (on ?t ?b1) 
                    (on ?t ?b2)
                    (on ?t ?b3)
                )
            )
        )
    )
    (preference secondLayerOfBlocks
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 - cube_block)
            (at-end 
                (and 
                    (on ?b1 ?b4) 
                    (on ?b2 ?b4)
                    (on ?b2 ?b5)
                    (on ?b3 ?b5)
                )
            )
        )
    )
    (preference thirdLayerOfBlocks
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
            (at-end 
                (and 
                    (on ?b1 ?b4) 
                    (on ?b2 ?b4)
                    (on ?b2 ?b5)
                    (on ?b3 ?b5)
                    (on ?b4 ?b6)
                    (on ?b5 ?b6)
                )
            )
        )
    )
    (preference mugOnTopOfPyrmaid
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block ?m - mug)
            (at-end 
                (and 
                    (on ?b1 ?b4) 
                    (on ?b2 ?b4)
                    (on ?b2 ?b5)
                    (on ?b3 ?b5)
                    (on ?b4 ?b6)
                    (on ?b5 ?b6)
                    (on ?b6 ?m)
                )
            )
        )
    )
    (preference dodgeballOnMug
        (exists (?m - mug ?d - dodgeball)
            (at-end 
                (and 
                    (on ?m ?d)
                )
            )
        )
    )
)) 
(:scoring maximize (+ 
    (count-once bookOnChairs)
    (* (count-once bookOnChairs) (count-once firstLayerOfBlocks))
    (* (count-once bookOnChairs) (count-once firstLayerOfBlocks) (count-once secondLayerOfBlocks))
    (* 
        (count-once bookOnChairs) (count-once firstLayerOfBlocks) 
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks)
    )
    (* 
        (count-once bookOnChairs) (count-once firstLayerOfBlocks) 
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks) 
        (count-once thirdLayerOfBlocks)
    )
    (* 
        (count-once bookOnChairs) (count-once firstLayerOfBlocks) 
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks) 
        (count-once thirdLayerOfBlocks) (count-once mugOnTopOfPyrmaid)
    )
    (* 
        (count-once bookOnChairs) (count-once firstLayerOfBlocks) 
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks) 
        (count-once thirdLayerOfBlocks) (count-once mugOnTopOfPyrmaid)
        (count-once dodgeballOnMug)
    )
)))


(define (game few-objects-28) (:domain few-objects-room-v1)
(:setup  
    (and 
        (forall (?b - cube_block)
            (game-optional (or 
                (on bed ?b)
                (exists (?c - cube_block) (and (not (= ?b ?c)) (on ?c ?b)))
            ))
        )
        (exists (?d - dodgeball) (game-optional (on bed ?d)))
    )
)
(:constraints (and 
    (preference allBlocksThrownToBinAndBallToChair
        (exists (?d - dodgeball ?h - hexagonal_bin ?c - chair)
            (then 
                (once (on bed agent))  ; with a conjunction with the setup if we wanted to allow multiple attempts
                (forall-sequence (?b - cube_block) 
                    (then 
                        (once (and (on bed agent) (agent_holds ?b)))
                        (hold (and (on bed agent) (not (agent_holds ?b)) (in_motion ?b))) 
                        (once (and (on bed agent) (on ?h ?b) (not (in_motion ?b))))
                        (hold (on bed agent))  ; until picking up the next cube
                    )
                )
                (once (and (on bed agent) (agent_holds ?d)))
                (hold-while 
                    (and (on bed agent) (not (agent_holds ?b)))
                    (touch ?c ?d)
                    (not (object_orientation ?c upright))
                ) 
            )
        )
    )
)) 
(:scoring maximize (+ 
    (* 0.5 (count-nonoverlapping allBlocksThrownToBinAndBallToChair))
)))


;29 is invalid


;30 is invalid, unless someone knows what "marabbalism" is ?


(define (game few-objects-31) (:domain few-objects-room-v1)
(:setup  

)
(:constraints (and 
    (preference blockThrownToGround
        (exists (?b cube_block)
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
            (hold-to-end (and (on floor ?b) (not (agent_holds ?b)) ))
        )
    )
    (preference blockThrownToBlock
        (exists (?b1 ?b2 - cube_block)
            (once (agent_holds ?b1))
            (hold (and (not (agent_holds ?b1)) (in_motion ?b1))) 
            (hold-to-end (and (on ?b1 ?b2) (not (agent_holds ?b1)) (not (agent_holds ?b2)) ))
        )
    )
)) 
(:scoring maximize (+ 
    (* (count-once-per-objects blockThrownToGround) 
       (+ 1 (count-once-per-objects blockThrownToBlock))
    )
)))
