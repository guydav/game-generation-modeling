; Key (1, 0, 3, 0, 0, 0, 0, 0, 2, 1, 0, 0)
(define (game evo-4088-182-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - cylindrical_block_blue)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - pyramid_block)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - cylindrical_block ?v1 - block ?v2 - pyramid_block_red)
        (at-end
          (and
            (on ?v0 ?v2)
            (on ?v1 ?v2)
            (on ?v1 ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4074-276-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (on ?v1 ?v0) (not (in_motion ?v0))))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v0 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0)
(define (game evo-4044-212-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near west_sliding_door ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - dodgeball ?v2 - hexagonal_bin)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v2 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - cylindrical_block ?v1 - pyramid_block_red ?v2 - cube_block)
        (at-end
          (and
            (on ?v0 ?v2)
            (on ?v0 ?v1)
            (on ?v2 ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (* -6 (count preference1) (count preference0))
)
)

; Key (1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0)
(define (game evo-4046-251-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near room_center ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - dodgeball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 20 (count preference0) (count preference1))
)
)

; Key (1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4083-155-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - ball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (not (in_motion ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* 15 (count preference0) (count preference1))
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4076-60-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - beachball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game evo-4081-135-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - ball ?v2 - hexagonal_bin)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v2 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in bottom_drawer ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference1))
)
)

; Key (1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4002-10-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near room_center ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - ball ?v2 - hexagonal_bin)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v2 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - doggie_bed ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (on ?v0 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game evo-4083-256-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in bottom_drawer ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4078-47-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bottom_drawer ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 14 (count preference1) (count preference0))
)
)

; Key (1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game evo-4095-194-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - ball)
    (game-optional
      (near desk ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects preference0)
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4074-277-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - hexagonal_bin ?v0 - basketball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v2 ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v2 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-4040-154-1) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (on desk ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4092-250-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (on ?v1 ?v0) (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference1))
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-4087-62-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball ?v2 - cylindrical_block_tan)
        (then
          (once (and (agent_holds ?v0) (adjacent ?v2 agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - game_object)
        (then
          (once (and (agent_holds ?v0) (adjacent side_table agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 17 (count preference2) (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4063-139-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near west_wall ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - ball ?v2 - hexagonal_bin)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v2 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bottom_drawer ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 7 (count preference0) (count preference1))
)
)

; Key (1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4063-77-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near west_sliding_door ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0) (on ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -8 (count preference0))
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4095-290-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game evo-4087-359-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects preference0)
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4085-208-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4052-22-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near west_wall ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - ball ?v2 - hexagonal_bin)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v2 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* 40 (count preference1) (count preference0))
)
)

; Key (1, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game evo-4036-364-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - dodgeball)
    (game-optional
      (near desk ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:terminal
  (>= (count preference0) 16)
)
(:scoring
  (* 14 (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0)
(define (game evo-4078-173-1) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - dodgeball)
    (game-optional
      (near desk ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on bed ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on floor ?v2)))
       )
     )
   )
 )
)
(:terminal
  (>= (count preference0) 16)
)
(:scoring
  (* 19 (count preference0) (count preference1))
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
(define (game evo-4074-284-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - cylindrical_block ?v1 - block ?v2 - pyramid_block_red)
        (at-end
          (and
            (on ?v0 ?v2)
            (on ?v1 ?v2)
            (on ?v1 ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0))
)
)

; Key (1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)
(define (game evo-4070-76-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - drawer)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v1)
           )
            (in bottom_drawer ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects preference0)
)
)

; Key (1, 0, 4, 0, 0, 2, 0, 0, 0, 0, 1, 0)
(define (game evo-4088-369-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - hexagonal_bin ?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - hexagonal_bin ?v2 - ball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v0 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* -7 (count preference0) (count preference2) (count preference3) (count preference1))
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4014-304-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v1 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0)
(define (game evo-4081-25-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - block)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - block)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -8 (count preference0) (count preference1))
)
)

; Key (1, 0, 3, 1, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game evo-4028-11-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near west_sliding_door ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (in ?v0 ?v1) (not (in_motion ?v1))))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -6 (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4030-89-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0) (on ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - ball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (not (in_motion ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference1) (count preference0))
)
)
