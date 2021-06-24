; 2 requires a lot of interpretation to figure out what the participant means
; since they refer to "figures" when it's unclear what that would be

; 3 seems impossible but valid, and also requires a fair bit of interpretation

(define (game many-objects-3) (:domain many-objects-room-v1)
(:setup (and
    (exists (?f - floor) (forall (?b - (either bridge_block flat_block)) (game-optional (on ?f ?b))))
))
(:constraints (and 
    (preference bounceBallToMug
        (exists (?g - golfball ?m - mug ?b - (either bridge_block flat_block)) 
            (then 
                ; ball starts in hand, with the agent on the chair, near the desk
                (once (and (agent_holds ?g) (on bed agent)))
                (hold-while
                    ; ball not in hand and in motion until...
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    ; the ball touches a block and then lands in/on the mug
                    (touch ?b ?g)
                ) 
                (once (and (on ?m ?g) (not (in_motion ?g))))
            )
        )
    )
))
(:scoring maximize (count-nonoverlapping bounceBallToMug)
))

; TODO: 5 is a juggling game - do we attempt to model it?

(define (game many-objects-4) (:domain many-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference twoBallsJuggled
        (exists (?g1 ?g2 - golfball) 
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
        )
    )
    ; the three ball case is even more complicated -- it's somethhing like:
    ; all three in hand => 1 in air => 1+2 in air => 2 in air => 2+3 in air => 3 in air => all three in hand
    (preference threeBallsJuggled
        (exists (?g1 ?g2 ?g3 - golfball)  
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
        )
    )
))
(:scoring maximize (+
    (* 10 (/ (count-longest threeBallsJuggled) 30))
    (* 5 (/ (count-longest twoBallsJuggled) 30))
    (* 100 (>= (count-longest threeBallsJuggled) 120))
    (* 50 (>= (count-longest twoBallsJuggled) 120))
)))


(define (game many-objects-5) (:domain many-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference agentOnRampOnEdge
        (exists (?r - large_triangular_ramp)
            (then
                (hold     
                    (and
                        (object_orientation ?r edge) 
                        (on ?r agent)
                    )   
                )
            )
        )
    )
))
(:scoring maximize (count-longest agentOnRampOnEdge)
))

; 6 is invalid


(define (game many-objects-7) (:domain many-objects-room-v1)
(:setup (and
    (exists (?r1 ?r2 - large_triangular_ramp) 
        (game-conserved (and
            (<= (distance ?r1 ?r2) 0.5)
        ))
    )
))
(:constraints (and 
    (preference circuit
        (exists (?r1 - large_triangular_ramp ?r2 - large_triangular_ramp ?c - chair ?h - hexagonal_bin ?b - beachball)
            (then 
                ; first, agent starts not between the ramps, then passes between them 
                ; so after not being between, it is between, then again not between
                (once (not (between ?r1 agent ?r2)))
                (any)
                (once (between ?r1 agent ?r2))
                (any)
                (once (not (between ?r1 agent ?r2)))
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
                (forall-sequence (?d - dodgeball)
                    (then
                        (once (agent_holds ?d))
                        (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                        (once (and (on ?h ?d) (not (in_motion ?d))))
                        (any)  ; to allow for a gap before the next dodgeball is picked up
                    )
                )
                ; bounce the beachball for 20 seconds
                (hold-for 20 (not (exists (?g - game_object) (or (on ?g ?b) (touch ?g ?b)))))
            )
        )
    )
))
(:scoring maximize (+
   (* 13 (count-once circuit))
   (* 2 (<= (count-shortest circuit) 60))
   (* 3 (<= (count-shortest circuit) 50))
   (* 2 (<= (count-shortest circuit) 40))
)))


(define (game many-objects-8) (:domain many-objects-room-v1)
(:setup (and
    (exists (?t1 ?t2 - tall_cylindrical_block ?r - curved_wooden_ramp ?h - hexagonal_bin) 
        (game-conserved (and
            (<= (distance ?t1 ?t2) 1)
            (= (distance ?r ?t1) (distance ?r ?t2))
            (adjacent_side ?h front ?r back)
            (= (distance ?h ?t1) (distance ?h ?t2))
            (< (distance ?r ?t1) (distance ?h ?t1))
        ))
    )
))
(:constraints (and 
    (preference throwBetweenBlocksToBin
        (exists (?g - golfball ?t1 ?t2 - tall_cylindrical_block ?r - curved_wooden_ramp ?h - hexagonal_bin)
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
                (once (and (on ?h ?g) (not (in_motion ?g))))
            ) 
        )
    )
    (preference thrownBallHitBlock
        (exists (?g - golfball ?t - tall_cylindrical_block) 
            (then
                ; ball starts in hand
                (once (agent_holds ?g))
                ; in motion, not in hand until...
                (hold (and (not (agent_holds ?g)) (in_motion ?g))) 
                ; the ball touches the block
                (once (touch ?g ?t)) 
            )
        ) 
    )
    (preference throwMissesBin
        (exists (?g - golfball ?h - hexagonal_bin)
            (then
                ; ball starts in hand
                (once (agent_holds ?g))
                ; ball not in hand and in motion until...
                (hold (and (not (agent_holds ?g)) (in_motion ?g)))
                ; ball settles and it's not in/on the bin
                (once (and (not (in_motion ?g)) (not (on ?h ?g))))
            )
        ) 
    )
    (preference throwAttempt
        (exists (?g - golfball)
            (then
                (once (agent_holds ?g))
                (hold (and (not (agent_holds ?g)) (in_motion ?g)))
                (once (not (in_motion ?g)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 15)
)
(:scoring maximize (+
    (* 5 (count-nonoverlapping throwBetweenBlocksToBin))
    (- (count-nonoverlapping thrownBallHitBlock))
    (- (* 2 (count-nonoverlapping throwMissesBin)))
)))


(define (game many-objects-9) (:domain many-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference throwBallToMugThroughRamp
        (exists (?g - golfball ?m - mug ?r - curved_wooden_ramp)  
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
        )
    )
    (preference throwBallToHexagonalBinThroughRamp
        (exists (?g - golfball ?h - hexagonal_bin ?r - curved_wooden_ramp) 
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
        )
    )
))
(:scoring maximize (+
    (* 5 (count-nonoverlapping throwBallToHexagonalBinThroughRamp))
    (* 10 (count-nonoverlapping throwBallToHexagonalBinThroughRamp))
))
)

; 10 is too ambiguous to resolve 

; 11 is invalid


(define (game many-objects-12) (:domain many-objects-room-v1)
(:setup (and
    (exists (?r - large_triangular_ramp ?h - hexagonal_bin) (forall (?b - block)
        ; to allow the agent to move them while playing?
        (or 
            (game-optional (adjacent ?b ?r))
            (game-optional (adjacent ?b ?h))
            (exists (?b2 - block)
                (game-optional (and 
                    (not (= ?b ?b2))
                    (or 
                        (on ?b2 ?b)
                        (adjacent ?b2 ?b)
                    )
                ))
            )
        )
    ))
))
(:constraints (and 
    (preference rollBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then 
                (once (agent_holds ?d))
                (hold (and 
                    (in_motion ?d) 
                    (not (agent_holds ?d))
                    (exists (?b - block) (touch ?d ?b))
                ))
                (once (and (on ?h ?d) (not (in_motion ?d))))
            )
        ) 
    ) 
))
(:scoring maximize (count-nonoverlapping rollBallToBin)  
))


; 13 requires counting within a preference


(define (game many-objects-14) (:domain many-objects-room-v1)
(:setup (and
    (exists (?m - mug ?h - hexagonal_bin ?f - floor ?r - rug ?b - bed ?d - desk)
        (game-conserved (and
            (not (on ?f ?r))
            (on ?f ?m)
            (on ?f ?h)
            (adjacent ?h ?m)
            (= (distance ?b ?h) (distance ?d ?h))
        ))
    )
))
(:constraints (and 
    (preference bounceBallToMug
        (exists (?g - golfball ?m - mug) 
            (then 
                (once (agent_holds ?g))
                (hold-while
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    (touch floor ?g)
                ) 
                (once (and (on ?m ?g) (not (in_motion ?g))))
            )
        )
    )
    (preference bounceBallToBin
        (exists (?g - golfball ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?g))
                (hold-while
                    (and (not (agent_holds ?g)) (in_motion ?g)) 
                    (touch floor ?g)
                ) 
                (once (and (on ?h ?g) (not (in_motion ?g))))
            )
        )
    )
))
(:scoring maximize (+
    (count-nonoverlapping bounceBallToMug)
    (* 3 (count-nonoverlapping bounceBallToBin))
)))


(define (game many-objects-15) (:domain many-objects-room-v1)
(:setup (and
    (forall (?b - (either cube_block tall_cylindrical_block short_cylindrical_block) ?r - room_center) 
        (game-optional (< (distance ?r ?b) 1))
    )
))
(:constraints (and 
    (preference blockOnFloor (exists (?b - (either cube_block tall_cylindrical_block short_cylindrical_block))
        (at-end
            (and 
                (in_building ?b)
                (on floor ?b)
            )
        )
    ))
    (preference blockOnBlock (exists (?b1 - (either cube_block tall_cylindrical_block short_cylindrical_block))
        (at-end
            (and 
                (in_building ?b1)
                (exists (?b2 - (either cube_block tall_cylindrical_block short_cylindrical_block)) (on ?b1 ?b2))
            )
        )
    )) 
))
(:scoring maximize (+ 
    (count-once blockOnFloor)
    (count-once-per-objects blockOnBlock)
)))


(define (game many-objects-16) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?r - large_triangular_ramp ?h - hexagonal_bin ?c - chair) 
        (game-conserved (and 
            (< (distance ?c ?r) 1)
            (> (distance ?r ?h) 1)
            (< (distance ?r ?h) 4)
            (between ?c ?r ?h)
        ))
    )
))
(:constraints (and 
    (preference dodgeballThrowAttempt
        (exists (?d - dodgeball)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )
    (preference dodgeballRollsOnRamp
        (exists (?d - dodgeball ?r - large_triangular_ramp ?c - chair)
            (then 
                (once (and (agent_holds ?d) (on ?c agent)))
                (hold-while (and (in_motion ?d) (not (agent_holds ?d))) 
                    (touch (side ?r front) ?d)
                    (touch (side ?r rear) ?d)
                )
            )
        ) 
    )
    (preference dodgeballOnRampToBin
        (exists (?d - dodgeball ?h - hexagonal_bin ?r - large_triangular_ramp ?c - chair)
            (then 
                (once (and (agent_holds ?d) (on ?c agent)))
                (hold-while (and (in_motion ?d) (not (agent_holds ?d))) 
                    (touch (side ?r front) ?d)
                    (touch (side ?r rear) ?d)
                )
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (on ?h ?d) (not (in_motion ?d))))
            )
        ) 
    )
    (preference golfballThrowAttempt
        (exists (?g - golfball)
            (then 
                (once (agent_holds ?g))
                (hold (and (not (agent_holds ?g)) (in_motion ?g))) 
                (once (not (in_motion ?g)))
            )
        )
    )
    (preference golfballRollsOnRamp
        (exists (?g - golfball ?r - large_triangular_ramp ?c - chair)
            (then 
                (once (and (agent_holds ?g) (on ?c agent)))
                (hold-while (and (in_motion ?g) (not (agent_holds ?g))) 
                    (touch (side ?r front) ?g)
                    (touch (side ?r rear) ?g)
                )
            )
        ) 
    )
    (preference golfballOnRampToBin
        (exists (?g - dodgeball ?h - hexagonal_bin ?r - large_triangular_ramp ?c - chair)
            (then 
                (once (and (agent_holds ?g) (on ?c agent)))
                (hold (and (in_motion ?g) (not (agent_holds ?g))))
                (hold-while (and (in_motion ?g) (not (agent_holds ?g))) 
                    (touch (side ?r front) ?g)
                    (touch (side ?r rear) ?g)
                )
                (hold (and (in_motion ?g) (not (agent_holds ?g))))
                (once (and (on ?h ?g) (not (in_motion ?g))))
            )
        ) 
    )
))
(:terminal (or 
    (>= (count-nonoverlapping dodgeballThrowAttempt) 3)
    (>= (count-nonoverlapping golfballThrowAttempt) 3)
    (and 
        (> (count-nonoverlapping dodgeballThrowAttempt) 0) 
        (> (count-nonoverlapping golfballThrowAttempt) 0) 
    )

))
(:scoring maximize (+
    (* 5 (count-nonoverlapping dodgeballRollsOnRamp) (= (count-nonoverlapping golfballThrowAttempt) 0) )
    (* 10 (count-nonoverlapping dodgeballOnRampToBin) (= (count-nonoverlapping golfballThrowAttempt) 0) )
    (* 5 (count-nonoverlapping golfballRollsOnRamp) (= (count-nonoverlapping dodgeballThrowAttempt) 0))
    (* 10 (count-nonoverlapping golfballOnRampToBin) (= (count-nonoverlapping dodgeballThrowAttempt) 0))
)))


(define (game many-objects-17) (:domain many-objects-room-v1)
(:setup
    (exists (?h - hexagonal_bin ?r - large_triangular_ramp)
        (game-conserved (adjacent (side ?h front) (side ?r back)))
    )
)
(:constraints (and 
    (preference throwBallBinThroughRamp
        (exists (?d - dodgeball ?h - hexagonal_bin ?r - large_triangular_ramp) 
            (then 
                (once (agent_holds ?d))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?d ?r)
                )
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
            )
        )
    )
))
(:scoring maximize (count-nonoverlapping throwBallBinThroughRamp)
))

; 18 is also ambiguous

; 19 is invalid


(define (game many-objects-20) (:domain many-objects-room-v1)
(:setup (and 
    (exists (?t1 ?t2 ?t3 - triangular_block)
        (game-conserved (and 
            (= (distance ?t1 ?t2) (distance ?t1 ?t3))
            (= (distance ?t1 ?t2) (distance ?t2 ?t3))
        ))
    )
    (forall (?b - bridge_block)
            (forall (?b2 - bridge_block ?t - triangular_block)
                (game-conserved (and
                    (not (= ?b ?b2))
                    (< (distance ?b ?t) (distance ?b ?b2))
                )
            )
        )
    )
    (exists (?r - large_triangular_ramp ?b - bridge_block ?t - triangular_block)
        (game-conserved (between ?r ?b ?t))
    )
))
(:constraints (and 
    (preference throwLandsInTarget
        (exists (?g - golfball ?r - large_triangular_ramp) 
            (then
                (once (and (agent_holds ?g) (exists (?t - triangular_block) (between agent ?r ?t))))
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?d ?g)
                )
                (once (and (not (in_motion ?g)) (on floor ?g) (exists (?t1 ?t2 - triangular_block) (between ?t1 ?r ?t2))))
            )
        )
    )
    (preference throwLandsInPerimeter
        (exists (?g - golfball ?r - large_triangular_ramp) 
            (then
                (once (and (agent_holds ?g) (exists (?t - triangular_block) (between agent ?r ?t))))
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?d ?g)
                )
                (once (and (not (in_motion ?g)) (on floor ?g) (exists (?b1 ?b2 - bridge_block) (between ?b1 ?r ?b2))))
            )
        )
    )
    (preference throwAttempt
        (exists (?g - golfball)
            (then 
                (once (agent_holds ?g))
                (hold (and (not (agent_holds ?g)) (in_motion ?g))) 
                (once (not (in_motion ?g)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping throwAttempt) 3)
)
(:scoring maximize (+
    (* 5 (count-nonoverlapping throwLandsInTarget))
    (* 2 (count-nonoverlapping throwLandsInPerimeter))
)))


(define (game many-objects-21) (:domain many-objects-room-v1)
(:setup (and
    (exists (?h - hexagonal_bin ?b - bed)
        (game-conserved (on ?b ?h))
    )
    (exists (?r - curved_wooden_ramp ?b - bed)
        (game-conserved (= (distance ?b ?r) 2))
    )
))
(:constraints (and 
    (preference throwBallToBinThroughRamp
        (exists (?d - dodgeball ?h - hexagonal_bin ?r - curved_wooden_ramp ?w - north_wall)  
            (then 
                ; assuming the bed is on the south wall
                (once (and (agent_holds ?d) (< (distance agent ?w) 1)))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?d ?r)
                )
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
            )
        )
    )
))
(:scoring maximize (count-nonoverlapping throwBallToBinThroughRamp)
))


(define (game many-objects-22) (:domain many-objects-room-v1)
(:setup
)
(:constraints (and 
    (preference beachballMovesSquareBlock
        (exists (?b - beachball ?c - (either flat_block bridge_block) ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference beachballMovesTallBlock
        (exists (?b - beachball ?c - (either pyramid_block tall_cylindrical_block) ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference beachballMovesShortBlock
        (exists (?b - beachball ?c - short_cylindrical_block ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference dodgeballMovesSquareBlock
        (exists (?b - dodgeball ?c - (either flat_block bridge_block) ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference dodgeballMovesTallBlock
        (exists (?b - dodgeball ?c - (either pyramid_block tall_cylindrical_block) ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference dodgeballMovesShortBlock
        (exists (?b - dodgeball ?c - short_cylindrical_block) 
            (then 
                (once (and (agent_holds ?b) (on floor ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference golfballMovesSquareBlock
        (exists (?b - golfball ?c - (either flat_block bridge_block) ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference golfballMovesTallBlock
        (exists (?b - golfball ?c - (either pyramid_block tall_cylindrical_block) ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
    (preference golfballMovesShortBlock
        (exists (?b - golfball ?c - short_cylindrical_block ?f - floor) 
            (then 
                (once (and (agent_holds ?b) (on ?f ?c)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (agent_holds ?c)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        )
    )
))
(:scoring maximize (+
   (* 1 (count-nonoverlapping beachballMovesSquareBlock))
   (* 3 (count-nonoverlapping beachballMovesTallBlock))
   (* 5 (count-nonoverlapping beachballMovesShortBlock))
   (* 2 (count-nonoverlapping dodgeballlMovesSquareBlock))
   (* 6 (count-nonoverlapping dodgeballMovesTallBlock))
   (* 10 (count-nonoverlapping dodgeballMovesShortBlock))
   (* 3 (count-nonoverlapping golfballMovesSquareBlock))
   (* 9 (count-nonoverlapping golfballMovesTallBlock))
   (* 15 (count-nonoverlapping golfballMovesShortBlock))
)))


(define (game many-objects-23) (:domain many-objects-room-v1)
(:setup
    (exists (?h - hexagonal_bin ?r1 ?r2 - large_triangular_ramp ?r - room_center)
        (game-conserved (and
             (< (distance ?r ?h) 0.5)
             (= (distance ?r1 ?r2) 0.67)
             (= (distance ?r1 ?h) 0.43)
            (= (distance ?r1 ?h) (distance ?r2 ?h))
        ))
    )
)
(:constraints (and 
    (preference beachballTolBinThroughRamp
        (exists (?b - beachball ?h - hexagonal_bin ?r - large_triangular_ramp) 
            (then 
                (once (agent_holds ?b))
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?r)
                )
                (once (and (on ?h ?b) (not (in_motion ?b)))) 
            )
        )
    )
    (preference dodgeballTolBinThroughRamp
        (exists (?d - dodgeball ?h - hexagonal_bin ?r - large_triangular_ramp) 
            (then 
                (once (agent_holds ?d))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?d ?r)
                )
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
            )
        )
    )
    (preference goflballTolBinThroughRamp
        (exists (?g - golfball ?h - hexagonal_bin ?r - large_triangular_ramp) 
            (then 
                (once (agent_holds ?g))
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?g ?r)
                )
                (once (and (on ?h ?g) (not (in_motion ?g)))) 
            )
        )
    )
    (preference throwBouncesOffRampAndMisses
        (exists (?b - (either beachball dodgeball golfball) ?h - hexagonal_bin ?r - large_triangular_ramp) 
            (then 
                (once (agent_holds ?b))
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?r)
                )
                (once (and (not (agent_holds ?b)) (not (on ?h ?b)) (not (in_motion ?b)))) 
            )
        )
    )
))
(:scoring maximize (+
   (* 10 (count-nonoverlapping beachballTolBinThroughRamp))
   (* 15 (count-nonoverlapping dodgeballTolBinThroughRamp))
   (* 20 (count-nonoverlapping goflballTolBinThroughRamp))
   (* (- 5) (count-nonoverlapping throwBouncesOffRampAndMisses))
)))


(define (game many-objects-24) (:domain many-objects-room-v1)
(:setup (and
    (exists (?h - hexagonal_bin) (forall (?b - block)
        ; to allow the agent to move them while playing?
        (or 
            (game-optional (adjacent ?b bed))
            (game-optional (adjacent ?b ?h))
            (exists (?b2 - block)
                (game-optional (and 
                    (not (= ?b ?b2))
                    (or 
                        (on ?b2 ?b)
                        (adjacent ?b2 ?b)
                    )
                ))
            )
        )
    ))
))
(:constraints (and 
    (preference rollBallToBin
        (exists (?g - golfball ?h - hexagonal_bin ?b - bed ?f - floor)
            (then 
                (once (and (agent_holds ?g) (on ?b agent)))
                (hold (and 
                    (in_motion ?g) 
                    (not (agent_holds ?g))
                    (exists (?b - block) (touch ?g ?b))
                ))
                (once (and 
                    (or (on ?h ?g) (on ?f ?g))
                    (not (in_motion ?g))
                ))
            )
        ) 
    )
    (preference rollAttempt
        (exists (?g - golfball ?b - block ?b - bed)
            (then 
                (once (and (agent_holds ?g) (on ?b agent)))
                (hold (and (in_motion ?g) (not (agent_holds ?g))))
                (once (touch ?g ?b))
            )
        ) 
    )
))
(:terminal
    (>= (count-nonoverlapping rollAttempt) 3)
)
(:scoring maximize (* 10 (count-total rollBallToBin))
))


(define (game many-objects-25) (:domain many-objects-room-v1)
(:setup (and 
    (exists (?t1 ?t2 - teddy_bear ?h - hexagonal_bin ?b - bed) (and  
        (game-conserved (on ?b ?t1))
        (game-conserved (on ?b ?t2))
        (game-conserved (on ?b ?h))
        ; player will hit them, so they might move
        (game-optional (= (distance ?t1 ?t2) 0.3))
        (game-optional (= (distance ?t1 ?h) (distance ?t2 ?h)))
    ))
    (forall (?c - chair ?d - desk)
        (game-conserved (> (distance ?c ?d) 2))
    )
    (forall (?b - (either dodgeball beachball))
        (game-optional (< (distance ?b ?d) 1))
    )
))
(:constraints (and 
    (preference ballHitsTeddyBear
        (exists (?b - (either dodgeball beachball) ?t - teddy_bear ?d - desk) 
            (then 
                (once (and (agent_holds ?b) (< (distance ?b ?d) 1)))
                (hold-while 
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?t)
                )
            )
        )
    )
    (preference ballLandsOnBin
        (exists (?b - (either dodgeball beachball) ?h - hexagonal_bin ?d - desk) 
            (then 
                (once (and (agent_holds ?b) (< (distance ?b ?d) 1)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?d) (not (in_motion ?d)))) 
            )
        )
    )
))
(:terminal 
    (>= (total-score) 1000)
)
(:scoring maximize (+
   (* 50 (count-nonoverlapping ballHitsTeddyBear))
   (* 100 (count-nonoverlapping ballLandsOnBin))
)))


(define (game many-objects-26) (:domain many-objects-room-v1)
(:setup (and 
    (exists (?s - shelf ?w - south_wall) (and 
        (game-conserved (adjacent ?s ?w))
        (exists (?h - hexagonal_bin) (game-conserved (< (distance ?s ?h) 0.1)))
        (forall (?b - (either tall_cylindrical_block bridge_block)) (game-optional (on ?s ?b)))
    ))
))
(:constraints (and 
    (preference golfballHitsLowerShelfBlocks
        (exists (?g - goflball ?b - (either triangular_block cube_block) ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?g))
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?g ?b)
                )
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference golfballHitsMiddleShelfBlocks
        (exists (?g - goflball ?b - (either pyramid_block short_cylindrical_block) ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?g))
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?g ?b)
                )
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
    (preference golfballHitsTopShelfBlocks
        (exists (?g - goflball ?b - (either flat_block tall_cylindrical_block bridge_block) ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?g))
                (hold-while 
                    (and (not (agent_holds ?g)) (in_motion ?g))
                    (touch ?g ?b)
                )
                (once (and (on ?h ?b) (not (in_motion ?b))))
            )
        )
    )
))
(:scoring maximize (+
   (* 1 (count-once-per-objects golfballHitsLowerShelfBlocks))
   (* 2 (count-once-per-objects golfballHitsMiddleShelfBlocks))
   (* 3 (count-once-per-objects golfballHitsTopShelfBlocks))
)))


(define (game many-objects-27) (:domain many-objects-room-v1)
(:setup (and
    (exists (?h - hexagonal_bin ?r - room_center) (game-conserved (< (distance ?r ?h) 0.5)))
    (exists (?e - bed) (forall (?b - (either golfball dodgeball beachball)) (game-optional (on ?e ?b))))
))
(:constraints (and 
    (preference golfballThrownToBin
        (exists (?b - golfball ?h - hexagonal_bin ?e - bed) 
            (then 
                (once (and (agent_holds ?b) (on ?e agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b)))) 
            )
        )
    )
    (preference dodgeballThrownToBin
        (exists (?b - dodgeball ?h - hexagonal_bin ?e - bed) 
            (then 
                (once (and (agent_holds ?b) (on ?e agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b)))) 
            )
        )
    )
    (preference beachballThrownToBin
        (exists (?b - beachball ?h - hexagonal_bin ?e - bed) 
            (then 
                (once (and (agent_holds ?b) (on ?e agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (on ?h ?b) (not (in_motion ?b)))) 
            )
        )
    )
))
(:scoring maximize (+
   (* 1 (count-nonoverlapping golfballThrownToBin))
   (* 2 (count-nonoverlapping dodgeballThrownToBin))
   (* 3 (count-nonoverlapping beachballThrownToBin))
)))


; 28 requires some accounting of "X satistfactions to preference A to move onto preference B"


(define (game many-objects-29) (:domain many-objects-room-v1)
(:setup (and
    (exists (?r - rug ?p - poster ?c - cd) (game-conserved (and 
        (on ?r ?p)
        (on ?p ?c)
    )))
))
(:constraints (and 
    (preference golfballThrown
        (exists (?g - golfball ?b - bed) 
            (then 
                (once (and (agent_holds ?g) (on ?b agent)))
                (hold (and (not (agent_holds ?g)) (in_motion ?g)))
                (hold-to-end (not (agent_holds ?g)))
            )
        )
    )
    (preference ballEndedOnCD
        (exists (?g - golfball ?c - cd) 
            (at-end (on ?c ?g))
        )
    )
    (preference ballEndedOnPoster
        (exists (?g - golfball) 
            (at-end (on poster ?g))
        )
    )
    (preference ballEndedOnRug
        (exists (?g - golfball) 
            (at-end (on rug ?g))
        )
    )
    (preference throwAttempt
        (exists (?g - golfball)
            (then 
                (once (agent_holds ?g))
                (hold (and (not (agent_holds ?g)) (in_motion ?g))) 
                (once (not (in_motion ?g)))
            )
        )
    )
))
(:terminal
    (>= (count-once-per-objects throwAttempt) 3)
)
(:scoring maximize (+
   (* 1 (count-nonoverlapping ballEndedOnRug))
   (* 2 (count-nonoverlapping ballEndedOnPoster))
   (* 3 (count-nonoverlapping ballEndedOnCD))
)))


(define (game many-objects-30) (:domain many-objects-room-v1)
(:setup
    (forall (?t - tall_cylindrical_block)
        (exists (?c - (either cube_block large_triangular_ramp))
            (game-optional (on ?c ?t)) ; game optional since player might fall and move
        )
    )
)
(:constraints (and 
    (preference agentOnBridge
        (exists (?t - tall_cylindrical_block) 
            (then 
                (hold (and 
                    (on ?t agent)
                    (exists (?c - (either cube_block large_triangular_ramp)) (on ?c ?t))
                ))
            )
        )
    )
))
(:scoring maximize (count-total agentOnBridge)
))



; 31 is invalid 