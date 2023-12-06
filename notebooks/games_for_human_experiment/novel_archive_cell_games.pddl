; Key (1, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0)
(define (game evo-4046-91-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - ball)
    (game-optional
      (near side_table ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - book ?v2 - desk)
        (at-end
          (and
            (on ?v2 ?v1)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - dodgeball)
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
  (* -4 (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-4032-65-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near door ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near floor ?v0)
         )
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
  (* -4 (count preference1) (count preference0))
)
)

; Key (1, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0)
(define (game evo-4083-166-0) (:domain medium-objects-room-v1)
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
      (exists (?v0 - alarm_clock ?v1 - hexagonal_bin)
        (at-end
          (in ?v1 ?v0)
       )
     )
   )
    (preference preference1
      (exists (?v0 - cylindrical_block ?v2 - cube_block ?v1 - cube_block)
        (at-end
          (and
            (on ?v2 ?v1)
            (on ?v0 ?v2)
         )
       )
     )
   )
 )
)
(:scoring
  (* 20 (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-4071-46-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - triangle_block_green)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - ball ?v2 - hexagonal_bin)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v2 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-3864-275-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball)
        (then
          (once (and (agent_holds ?v0) (adjacent ?v0 agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:terminal
  (>= (count-once preference1) 1)
)
(:scoring
  (* 19 (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4028-365-0) (:domain medium-objects-room-v1)
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
          (once (and (not (in_motion ?v2)) (on bottom_drawer ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* 7 (count preference0) (count preference1))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2)
(define (game evo-4069-281-1) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near room_center ?v0)
         )
       )
     )
   )
    (preference preference1
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
  (+ (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4068-116-0) (:domain medium-objects-room-v1)
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
      (exists (?v2 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (in ?v1 ?v2) (on ?v1 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* 70 (count preference1) (count preference0))
)
)

; Key (1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 0)
(define (game evo-4077-14-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - ball)
    (game-optional
      (near bed ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
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
    (preference preference1
      (exists (?v0 - ball)
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
  (* 20 (count preference0) (count preference1))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0)
(define (game evo-4058-43-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - book ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v2 - triangle_block_green)
        (then
          (once (agent_holds ?v2))
          (hold (not (agent_holds ?v2)))
          (hold (agent_holds ?v2))
          (once (not (in_motion ?v2)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0)
(define (game evo-4074-308-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 ?v1 - game_object)
        (at-end
          (in ?v0 ?v1)
       )
     )
   )
    (preference preference1
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (agent_holds ?v0)
           )
            (not
              (in_motion ?v0)
           )
            (in bottom_drawer ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (* 80 (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0)
(define (game evo-4082-215-1) (:domain medium-objects-room-v1)
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
          (once (and (not (in_motion ?v1)) (on floor ?v1)))
       )
     )
   )
 )
)
(:terminal
  (>= (count preference1) 16)
)
(:scoring
  (* 19 (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-4084-365-0) (:domain medium-objects-room-v1)
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
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* 40 (count preference0) (count preference1))
)
)

; Key (1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2)
(define (game evo-4089-277-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near door ?v0)
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
            (near room_center ?v1)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near door ?v0)
         )
       )
     )
   )
    (preference preference2
      (exists (?v0 - ball ?v1 - hexagonal_bin)
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
  (* -4 (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0)
(define (game evo-4071-253-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - pyramid_block_red)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - tall_cylindrical_block)
        (then
          (once (agent_holds ?v1))
          (hold (not (agent_holds ?v1)))
          (hold (agent_holds ?v1))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - triangle_block_green)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 1, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-4037-266-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - book ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference2
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
  (* -6 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 1, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0)
(define (game evo-3965-117-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - basketball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
    (preference preference2
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
  (* -4 (count preference1) (count preference2) (count preference0))
)
)

; Key (1, 1, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4094-364-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0) (on ?v1 ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference2
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
  (* -4 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0)
(define (game evo-4019-48-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v0 ?v1) (not (in_motion ?v1))))
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
    (preference preference2
      (exists (?v1 - cylindrical_block ?v0 - block ?v2 - pyramid_block_red)
        (at-end
          (and
            (on ?v1 ?v2)
            (on ?v1 ?v0)
            (on ?v0 ?v2)
         )
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 1, 3, 0, 0, 0, 1, 0, 0, 1, 1, 0)
(define (game evo-4080-173-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - mug ?v2 - hexagonal_bin)
        (at-end
          (in ?v2 ?v1)
       )
     )
   )
    (preference preference2
      (exists (?v1 - block ?v3 - block ?v0 - cube_block_blue)
        (at-end
          (and
            (on ?v1 ?v0)
            (on ?v3 ?v0)
            (on ?v3 ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (* -8 (count preference1) (count preference2) (count preference0))
)
)

; Key (1, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1)
(define (game evo-4071-118-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball ?v1 - block)
        (then
          (once (and (agent_holds ?v0) (adjacent ?v1 agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - game_object)
        (then
          (once (and (agent_holds ?v0) (adjacent bed agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 20 (count preference2) (count preference0) (count preference1))
)
)

; Key (1, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-4094-333-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - cube_block_blue)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
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
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference2) (count preference0) (count preference1))
)
)

; Key (1, 0, 3, 2, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-4055-14-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - ball)
    (game-optional
      (near bed ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - book ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - dodgeball_blue)
        (then
          (once (and (agent_holds ?v0) (adjacent ?v0 agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 1, 4, 0, 0, 0, 0, 0, 0, 2, 2, 0)
(define (game evo-4055-271-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (not (in ?v0 ?v1))))
       )
     )
   )
    (preference preference1
      (exists (?v2 - hexagonal_bin ?v0 - dodgeball_red)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v2 ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - cube_block ?v1 - block ?v2 - pyramid_block_red)
        (at-end
          (and
            (on ?v1 ?v2)
            (on ?v0 ?v1)
         )
       )
     )
   )
    (preference preference3
      (exists (?v1 - cylindrical_block ?v0 - block ?v3 - pyramid_block_red)
        (at-end
          (and
            (on ?v1 ?v3)
            (on ?v0 ?v3)
            (on ?v1 ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (* -10 (count preference3) (count preference1) (count preference0) (count preference2))
)
)

; Key (1, 0, 4, 0, 1, 0, 0, 2, 0, 0, 0, 0)
(define (game evo-4037-225-0) (:domain medium-objects-room-v1)
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
      (exists (?v1 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v1)
           )
            (on desk ?v1)
         )
       )
     )
   )
    (preference preference1
      (exists (?v2 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v2)
           )
            (on bed ?v2)
         )
       )
     )
   )
    (preference preference2
      (exists (?v1 - hexagonal_bin ?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0) (on ?v1 ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - hexagonal_bin ?v3 - dodgeball)
        (then
          (once (agent_holds ?v3))
          (hold (and (not (agent_holds ?v3)) (in_motion ?v3)))
          (once (and (not (in_motion ?v3)) (on ?v0 ?v3)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference0) (count preference3) (count preference2) (count preference1))
)
)

; Key (1, 1, 4, 0, 0, 1, 1, 0, 2, 0, 0, 0)
(define (game evo-4064-240-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - triangle_block_green)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - cylindrical_block)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v1 - beachball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on bed ?v1)))
       )
     )
   )
    (preference preference3
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
  (* -4 (count preference3) (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-4094-300-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near room_center ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - game_object)
        (then
          (once (and (agent_holds ?v0) (same_color ?v0 white)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - dodgeball ?v1 - hexagonal_bin)
        (at-end
          (in ?v1 ?v0)
       )
     )
   )
    (preference preference3
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
  (* -6 (count preference0) (count preference2) (count preference1) (count preference3))
)
)

; Key (1, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-3971-172-1) (:domain medium-objects-room-v1)
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
          (once (and (on ?v0 ?v1) (not (in_motion ?v1))))
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
    (preference preference2
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bottom_drawer ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - hexagonal_bin ?v1 - golfball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference2) (count preference1) (count preference3))
)
)

; Key (1, 0, 4, 0, 1, 0, 0, 1, 0, 2, 0, 0)
(define (game evo-4085-272-1) (:domain medium-objects-room-v1)
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
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v2)
           )
            (on bed ?v2)
         )
       )
     )
   )
    (preference preference2
      (exists (?v0 - cylindrical_block ?v1 - pyramid_block_red ?v3 - pyramid_block_yellow)
        (at-end
          (and
            (on ?v0 ?v3)
            (on ?v0 ?v1)
         )
       )
     )
   )
    (preference preference3
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
  (* -2 (count preference0) (count preference3) (count preference2) (count preference1))
)
)

; Key (1, 1, 4, 0, 0, 1, 0, 0, 0, 0, 1, 0)
(define (game evo-4012-319-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on desk ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on bottom_drawer ?v1)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - ball ?v1 - hexagonal_bin)
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
  (* -4 (count preference1) (count preference2) (count preference0) (count preference3))
)
)

; Key (1, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 2)
(define (game evo-4009-365-1) (:domain medium-objects-room-v1)
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
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v2)
           )
            (near room_center ?v2)
         )
       )
     )
   )
    (preference preference3
      (exists (?v2 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v2)
           )
            (near floor ?v2)
         )
       )
     )
   )
 )
)
(:scoring
  (* 0.7 (count preference0) (count preference3) (count preference1) (count preference2))
)
)

; Key (1, 1, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-4079-220-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - cube_block)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - ball ?v3 - hexagonal_bin)
        (at-end
          (in ?v3 ?v2)
       )
     )
   )
    (preference preference3
      (exists (?v1 - cylindrical_block ?v0 - pyramid_block_red ?v2 - cube_block)
        (at-end
          (and
            (on ?v1 ?v2)
            (on ?v1 ?v0)
            (not
              (agent_holds ?v1)
           )
         )
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference0) (count preference2) (count preference1) (count preference3))
)
)

; Key (1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4086-7-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bottom_drawer ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (on bottom_drawer ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (at-end
          (in ?v1 ?v0)
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference1) (count preference3) (count preference0) (count preference2))
)
)
