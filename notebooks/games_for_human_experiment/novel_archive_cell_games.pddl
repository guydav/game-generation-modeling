; Key (1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-4063-115-1) (:domain medium-objects-room-v1)
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
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-4029-119-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - pyramid_block_red)
    (game-conserved
      (on floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - block)
        (then
          (once (agent_holds ?v1))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1))))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects preference0)
)
)

; Key (1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)
(define (game evo-4034-217-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (in bottom_drawer ?v0)
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

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-3940-26-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - chair)
        (then
          (once (and (agent_holds ?v0) (adjacent side_table agent)))
          (hold (and (not (agent_holds ?v0)) (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4050-288-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
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
  (count preference0)
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
(define (game evo-4019-113-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - cylindrical_block)
    (game-conserved
      (on floor ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
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
  (count-once-per-objects preference0)
)
)

; Key (1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4038-40-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - beachball ?v1 - hexagonal_bin)
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
  (count preference0)
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-4069-100-0) (:domain medium-objects-room-v1)
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
 )
)
(:scoring
  (count-once-per-objects preference0)
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-4051-41-0) (:domain medium-objects-room-v1)
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
      (exists (?v0 - ball ?v1 - chair)
        (then
          (once (and (agent_holds ?v0) (adjacent desk agent)))
          (hold (and (not (agent_holds ?v0)) (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4084-34-0) (:domain medium-objects-room-v1)
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
  (count preference0)
)
)

; Key (1, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0)
(define (game evo-4089-169-1) (:domain medium-objects-room-v1)
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
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on bed ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-4063-325-1) (:domain medium-objects-room-v1)
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
          (once (and (not (in_motion ?v0)) (on bottom_drawer ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0)
(define (game evo-4088-59-1) (:domain medium-objects-room-v1)
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
      (exists (?v2 - key_chain ?v3 - hexagonal_bin)
        (at-end
          (in ?v3 ?v2)
       )
     )
   )
 )
)
(:scoring
  (* 0 (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0)
(define (game evo-4026-65-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - block)
    (game-conserved
      (near west_wall ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - cylindrical_block ?v2 - block ?v3 - pyramid_block_red)
        (at-end
          (and
            (on ?v1 ?v3)
            (on ?v2 ?v1)
            (on ?v2 ?v3)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - cylindrical_block ?v1 - block ?v2 - pyramid_block_red)
        (at-end
          (and
            (on ?v0 ?v2)
            (on ?v1 ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (* 3 (count preference0) (count preference1))
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

; Key (1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4092-254-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near top_shelf ?v0)
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
          (once (and (on ?v1 ?v0) (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - ball ?v1 - hexagonal_bin)
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
  (* -4 (count preference1) (count preference0))
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

; Key (1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0)
(define (game evo-3968-207-1) (:domain medium-objects-room-v1)
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
            (on side_table ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v1 - game_object)
        (at-end
          (and
            (in bottom_drawer ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (* 20 (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1)
(define (game evo-4060-102-0) (:domain medium-objects-room-v1)
(:setup
  (forall (?v0 - dodgeball)
    (game-optional
      (near bed ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - chair)
        (then
          (once (and (agent_holds ?v0) (adjacent side_table agent)))
          (hold (and (not (agent_holds ?v0)) (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - cube_block)
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
  (* 40 (count preference0) (count preference1))
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

; Key (1, 1, 3, 0, 0, 1, 1, 0, 0, 0, 1, 0)
(define (game evo-4042-47-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in bottom_drawer ?v0)))
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
      (exists (?v0 - game_object)
        (at-end
          (and
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
  (* 4 (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 0, 3, 1, 0, 0, 0, 1, 1, 0, 0, 0)
(define (game evo-4087-330-1) (:domain medium-objects-room-v1)
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
      (exists (?v0 - teddy_bear ?v1 - desk)
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
      (exists (?v0 - tall_cylindrical_block)
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
  (* 0.1 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2)
(define (game evo-3919-279-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - chair)
        (then
          (once (and (agent_holds ?v0) (adjacent side_table agent)))
          (hold (and (not (agent_holds ?v0)) (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - triangle_block_tan)
        (then
          (once (and (agent_holds ?v0) (adjacent ?v0 agent)))
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
  (* 10 (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4035-274-0) (:domain medium-objects-room-v1)
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
      (exists (?v1 - hexagonal_bin ?v0 - golfball_orange)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (in ?v1 ?v0) (not (in_motion ?v0)) (on ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (in ?v0 ?v2) (on ?v0 ?v2)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - ball ?v2 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (on ?v2 ?v0) (not (in_motion ?v0)) (in ?v2 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0)
(define (game evo-4005-342-1) (:domain medium-objects-room-v1)
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
      (exists (?v0 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v0 ?v2)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - cylindrical_block ?v1 - pyramid_block_red ?v2 - pyramid_block_red)
        (at-end
          (and
            (on ?v0 ?v2)
            (on ?v0 ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (* 3 (count preference1) (count preference0) (count preference2))
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0)
(define (game evo-4010-15-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - basketball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
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
    (preference preference2
      (exists (?v0 - ball ?v2 - hexagonal_bin)
        (then
          (once (and (agent_holds ?v0) (adjacent ?v0 agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (in ?v2 ?v0) (on ?v2 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0, 1)
(define (game evo-3897-241-1) (:domain medium-objects-room-v1)
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
      (exists (?v2 - cylindrical_block ?v0 - cylindrical_block ?v3 - cube_block_blue ?v1 - cube_block)
        (at-end
          (and
            (on ?v2 ?v1)
            (on ?v0 ?v1)
            (on ?v1 ?v3)
         )
       )
     )
   )
 )
)
(:scoring
  (* 2 (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 1, 3, 0, 0, 2, 0, 1, 0, 0, 0, 0)
(define (game evo-4053-248-0) (:domain medium-objects-room-v1)
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
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on desk ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* 19 (count preference1) (count preference0) (count preference2))
)
)

; Key (1, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0)
(define (game evo-4079-12-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (and (agent_holds ?v1) (adjacent ?v1 agent)))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - alarm_clock ?v2 - hexagonal_bin)
        (at-end
          (in ?v2 ?v0)
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
  (* -7 (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4052-143-1) (:domain medium-objects-room-v1)
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
          (once (and (agent_holds ?v0) (not (in_motion ?v0))))
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
          (once (and (agent_holds ?v0) (adjacent ?v0 agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 1, 4, 0, 0, 2, 1, 0, 0, 1, 0, 0)
(define (game evo-4079-275-1) (:domain medium-objects-room-v1)
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
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on desk ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - key_chain ?v0 - hexagonal_bin)
        (at-end
          (in ?v0 ?v2)
       )
     )
   )
    (preference preference3
      (exists (?v0 - cylindrical_block ?v2 - pyramid_block ?v1 - pyramid_block_red)
        (at-end
          (and
            (on ?v2 ?v1)
            (on ?v0 ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (* -4 (count preference1) (count preference2) (count preference0) (count preference3))
)
)

; Key (1, 0, 4, 0, 1, 0, 0, 0, 1, 0, 1, 1)
(define (game evo-4085-32-1) (:domain medium-objects-room-v1)
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
    (preference preference1
      (exists (?v1 - flat_block_yellow)
        (then
          (once (agent_holds ?v1))
          (hold (not (agent_holds ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (on ?v1 ?v2) (not (in_motion ?v2)) (in ?v1 ?v2)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* 2 (count preference3) (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4073-131-0) (:domain medium-objects-room-v1)
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
  (* 0 (count preference0) (count preference2) (count preference1) (count preference3))
)
)

; Key (1, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0)
(define (game evo-4091-177-1) (:domain medium-objects-room-v1)
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
          (once (and (not (in_motion ?v0)) (on bottom_drawer ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v2 - ball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v0 ?v2)))
       )
     )
   )
    (preference preference3
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
  (* 9 (count preference0) (count preference3) (count preference1) (count preference2))
)
)

; Key (1, 0, 4, 0, 0, 0, 2, 0, 0, 0, 1, 0)
(define (game evo-4063-286-1) (:domain medium-objects-room-v1)
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
      (exists (?v0 ?v1 - game_object)
        (at-end
          (and
            (in ?v0 ?v1)
         )
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
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - game_object)
        (at-end
          (and
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
  (* -4 (count preference0) (count preference2) (count preference1) (count preference3))
)
)

; Key (1, 0, 4, 0, 0, 3, 0, 0, 0, 0, 0, 0)
(define (game evo-4001-229-0) (:domain medium-objects-room-v1)
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
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - dodgeball_pink)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
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
    (preference preference3
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (agent_holds ?v1) (on desk ?v1)))
       )
     )
   )
 )
)
(:scoring
  (* 2 (count preference3) (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 1, 4, 0, 1, 0, 0, 0, 3, 0, 0, 0)
(define (game evo-4071-327-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - tall_cylindrical_block_yellow)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - flat_block_yellow)
        (then
          (once (agent_holds ?v1))
          (hold (not (agent_holds ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (in ?v2 ?v1) (not (in_motion ?v1)) (on ?v2 ?v1)))
       )
     )
   )
    (preference preference3
      (exists (?v2 - pyramid_block_yellow)
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
  (* 20 (count preference3) (count preference0) (count preference1) (count preference2))
)
)

; Key (1, 0, 4, 0, 0, 0, 0, 2, 0, 1, 0, 0)
(define (game evo-4085-305-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near rug ?v0)
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
      (exists (?v0 - book ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v0 ?v2)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - cylindrical_block ?v2 - block ?v1 - pyramid_block_red)
        (at-end
          (and
            (on ?v0 ?v1)
            (on ?v2 ?v1)
            (on ?v2 ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (* 20 (count preference1) (count preference0) (count preference2) (count preference3))
)
)

; Key (1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-4074-207-1) (:domain medium-objects-room-v1)
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
      (exists (?v1 - laptop)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on bed ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - ball ?v3 - hexagonal_bin)
        (at-end
          (in ?v3 ?v2)
       )
     )
   )
    (preference preference2
      (exists (?v1 - laptop)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (in_motion ?v1))
       )
     )
   )
    (preference preference3
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
  (* -6 (count preference3) (count preference2) (count preference0) (count preference1))
)
)

; Key (1, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-4051-69-1) (:domain medium-objects-room-v1)
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
      (exists (?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
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
      (exists (?v0 - dodgeball)
        (then
          (once (and (agent_holds ?v0) (adjacent ?v0 agent)))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference3
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
  (* 17 (count preference0) (count preference3) (count preference1) (count preference2))
)
)
