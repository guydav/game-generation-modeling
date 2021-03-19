; create tunnels with the brigde blocks and throw the balls
; "1 point for teach"

(define (game medium-objects-3) (:domain medium-objects-room-v1)
(:setup (and
    (forall (?f - flat_block) (game-conserved (on floor ?f)))
    (forall (?b - bridge_block) 
        (exists (?f - flat_block) 
            (game-conserved (on ?f ?b))
        )
    )
))
(:constraints (and 
    (preference throwBallUnderBridge
        (exists (?b - basketball ?bb - bridge_block)  
            (then 
                ; ball starts in hand, not under the bridge
                (once (and (agent_holds ?b) (not (under ?bb ?b))))
                ; neither ball nor block in hand until...
                (hold-while 
                    (and (in_motion ?b) (not (agent_holds ?b)) (not (agent_holds ?bb)))
                    (under ?bb ?b)
                )     
                ; the ball is under the bridge and then again not under the bridge
                (once (not (under ?bb ?b))) 
                ) 
            )
        ) 
    ) )
) )
(:metric maximize (count-nonoverlapping throwBallUnderBridge)) 
)


(define (game medium-objects-4) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?s - shelf ?h - hexagonal_bin) 
        (and
            (game-conserved (on ?s ?h))
            (game-conserved (forall (?s2 - shelf) (>= (distance ?s desk) (distance ?s2 desk))))
        )
    )
))
(:constraints (and 
    (preference throwBallFromChairToBin
        (exists (?b - basketball ?c - chair ?h - hexagonal_bin) 
            (then
                ; ball starts in hand, with the agent on the chair, near the desk
                (once (and (agent_holds ?b) (on ?c agent) (adjacent ?c desk) (agent_perspective upside_down)))
                ; ball not in hand until...
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                ; the ball is in the bin
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        ) ) 
    ))
) )
(:scoring maximize (count-nonoverlapping throwBallFromChairToBin))
)


(define (game medium-objects-5) (:domain medium-objects-room-v1)
; TODO: move building definition(s) to the domain
; (:objects  ; we'd eventually populate by script
;     tower - building  
; )
(:setup (and
    (game-optional (forall (?b - block) (on floor ?b)))
))
(:constraints (and
    ; Here we have the preference before the quantifier, to count it at most once
    (preference blockOnFloor (exists (?b - block) 
        (then 
            (hold
                (and
                    (on floor ?b)
                    (in_building tower ?b)
                )
            )
            (once (building_fell tower))
        )
    ))
    ; Here we have the quantifier before, to count how many times it happens 
    (preference blockOnBlock (exists (?b - block ?b2 - block)
        (then
            (hold
                (and 
                    ; both blocks are in the tower
                    (in_building tower ?b)
                    (in_building tower ?b2)
                    ; this new block ?b is on top of the second block ?b2
                    (on ?b ?b2) ; an object cannot be on itself, so this fails if ?b = ?b2
                )
            )
            ; until the tower falls
            (once (building_fell tower))
        )
    ))
    (preference blockFellNear (exists (?b - block) 
        (then
            ; block is in the towr until
            (hold (in_building ?b)
            ; starting with the building falling
            (once (building_fell tower))
            ; block is falling without agent moving it until -- this only works if the blocks start moving the state after the previous state happens
            (hold (and (not (agent_holds ?b) (in_motion ?b)))) 
            ; it settles near the tower
            (once (<= (distance tower ?b) 0.1)) 
        )
    )) 
))
(:metric maximize (+
    (count-once blockOnFloor)
    (count-once-per-objects blockOnBlock)
    (- (count-once-per-objects blockFellNear))
))
)

; 6 has no setup

(define (game medium-objects-6) (:domain medium-objects-room-v1)
; TODO: move this to the domain definition
; (:objects  ; we'd eventually populate by script
;     tower - building  
; )
(:constraints (and
    ; Count how many objects are part of the tower
    (preference objectInTower (exists (?o - game_object)
        (then
            (once (agent_holds ?b))
            (hold (in_building tower ?o))
            (once (building_fell tower))
        )
    ))
))
(:metric scoring (/ (* 100 (max_height tower)) (count-once-per-objects objectInTower))
)
)

;7 is invalid

(define (game scoring-8) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?h - hexagonal_bin) (exists (?d - doggie_bed) (exists (?p - pillow)
        (game-conserved (and
                (<= (distance ?h ?d) 1)
                (<= (distance ?h ?p) 1)
                (<= (distance ?p ?d) 1)
                (< (distance agent ?h) (distance agent ?d))
                (< (distance agent ?h) (distance agent ?p))
                (< (distance agent ?d) (distance agent ?p))
        ))
    )))
))
(:constraints (and
    (preference beachballToHexagonalBin
        (exists (?b - beachball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference beachballToDoggieBed
        (exists (?b - beachball ?d - doggie_bed)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?d ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference beachballToPillow
        (exists (?b - beachball ?p - pillow) 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?p ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference dodgeballToHexagonalBin
        (exists (?b - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference dodgeballToDoggieBed
        (exists (?b - dodgeball ?d - doggie_bed)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?d ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference dodgeballToPillow
        (exists (?b - dodgeball ?p - pillow) 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?p ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference basketballToHexagonalBin
        (exists (?b - basketball ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference basketballToDoggieBed
        (exists (?b - basketball ?d - doggie_bed)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?d ?b) (not (in_motion ?b))))
            )
        ))
    )
    (preference basketballToPillow
        (exists (?b - basketball ?p - pillow)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?p ?b) (not (in_motion ?b))))
            )
        ))
    )
))
(:metric maximize (+ 
    (* 3 (count-nonoverlapping beachballToHexagonalBin))
    (* 5 (count-nonoverlapping beachballToDoggieBed))
    (* 7 (count-nonoverlapping beachballToPillow))
    (* 6 (count-nonoverlapping dodgeballToHexagonalBin))
    (* 8 (count-nonoverlapping dodgeballToDoggieBed))
    (* 10 (count-nonoverlapping dodgeballToPillow))
    (* 9 (count-nonoverlapping basketballToHexagonalBin))
    (* 11 (count-nonoverlapping basketballToDoggieBed))
    (* 13 (count-nonoverlapping basketballToPillow))
))
)

(define (game scoring-9) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?t1 - tall_cylindrical_block ?t2 - tall_cylindrical_block ?tb - teddy_bear)
        (game-conserved (and
            (not (= ?t1 ?t2))
            (<= (distance ?t1 ?t2) 2)
            (= (distance ?tb ?t1) (distance ?tb ?t2))
        ))
    )
))
(:constraints (and 
    ; TODO: is the subject refers to it first as throwing, and then as rolling, should we consider it?
    (preference throwBetweenBlocksToBear
        (exists (?b - basketball ?t1 - tall_cylindrical_block ?t2 - tall_cylindrical_block ?tb - teddy_bear)
            (then 
                (once (agent_holds ?b))
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (between ?t1 ?b ?t2)
                )
                (once (touch ?b ?tb))
            )
        ) 
    )
    (preference thrownBallHitBlock
        (exists (?b - basketball ?t - tall_cylindrical_block) 
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))block
                (once (touch ?b ?t)) 
            )
        ) 
    )
))
(:metric maximize (+
    (* 15 (count-nonoverlapping throwBetweenBlocksToBear))
    (* (- 5) (count-nonoverlapping thrownBallHitBlock))
))

; 10 has no setup

(define (game medium-objects-10) (:domain medium-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference thrownObjectKnocksDesktop
        (exists (?o - (either pillow beachball dodgeball) ?d - desktop)
            (then
                ; starts with agent holding the desktop
                (once (agent_holds ?o))
                (hold-while
                    ; while the object is being thrown and the agent is touching neither the object nor the desktop
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?d)))
                    ; the thrown object hits the desktop
                    (touch ?o ?d)
                )
                ; eventually knocking it off the desk
                (once (and (not (on desk ?d)) (not (in_motion ?d)))) 
            )
        ) 
    )
    (preference thrownObjectKnocksDeskLamp
        (exists (?o - (either pillow beachball dodgeball) ?d - desk_lamp)
            (then
                ; starts with agent holding the desktop
                (once (agent_holds ?o))
                (hold-while
                    ; while the object is being thrown and the agent is touching neither the object nor the desk lamp
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?d)))
                    ; the thrown object hits the desk lamp
                    (touch ?o ?d)
                )
                ; eventually knocking it off the desk
                (once (and (not (on desk ?d)) (not (in_motion ?d)))) 
            )
        ) 
    )
    (forall (preference thrownObjectKnocksCD
        (exists (?o - (either pillow beachball dodgeball) ?c - cd)
            (then
                ; starts with agent holding the desktop
                (once (agent_holds ?o))
                (hold-while
                    ; while the object is being thrown and the agent is touching neither the object nor the cd
                    (and (not (agent_holds ?o)) (in_motion ?o) (not (agent_holds ?c)))
                    ; the thrown object hits the cd
                    (touch ?o ?c)
                )
                ; eventually knocking it off the desk
                (once (and (not (on desk ?c)) (not (in_motion ?c)))) 
            )
        ) 
    ))
))
(:scoring maximize (+
    (* 5 (count-nonoverlapping thrownObjectKnocksDesktop))
    (* 10 (count-nonoverlapping thrownObjectKnocksDeskLamp))
    (* 15 (count-nonoverlapping thrownObjectKnocksCD))
))

; 11 has no setup

(define (game scoring-11) (:domain medium-objects-room-v1)
(:objects  ; we'd eventually populate by script
)
(:init ; likewise - we could populate fully by a script
)
(:constraints (and 
    (forall (?b - basketball) (preference throwBallWithEyesClosed
        (exists (?h - hexagonal_bin) 
            (then
                ; ball starts in hand, with the agent on the chair, near the desk
                (once (and (agent_holds ?b) (agent_perspective eyes_closed)))
                ; ball not in hand and in motion until...
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                ; the ball is in the bin
                (once (and (on ?h ?b) (not (in_motion ?b))))
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
(:metric maximize (* 5(is-violated throwBallFromChairToBin)))
)

; TODO: 12 has no setup, and is a little nonsensical, but could be modeled like this:

(define (game medium-objects-12) (:domain medium-objects-room-v1)
; TODO: move to the domain decleration
; (:objects  ; we'd eventually populate by script
;     castle - building  
; )
(:constraints (and
    (preference correctColorBlock (exists (?b - block) 
        (at-end
            (and 
                (in_building castle ?b))
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
(:scoring maximize (* (2 (count-once-per-objects correctColorBlock)))
)
)

