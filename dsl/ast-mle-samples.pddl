
(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference preference2
      (exists (?y - (either dodgeball yellow))
        (exists (?u - dodgeball ?e - (either wall golfball))
          (exists (?j - hexagonal_bin)
            (exists (?d - dodgeball)
              (then
                (once (and (and (in_motion bed) (not (not (on agent) ) ) ) ) )
                (hold (in_motion ?j) )
              )
            )
          )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (and (not (touch agent ?h) ) (not (agent_holds ?h) ) ) )
          (once (not (object_orientation ?h upright) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - hexagonal_bin)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (and (not (touch agent ?h) ) (not (agent_holds ?h) ) ) )
          (once (not (object_orientation ?h upright) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (and (not (touch agent ?h) ) (not (in_motion north_wall) ) ) )
          (once (not (object_orientation ?h upright) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference preference2
      (then
        (hold (and (agent_holds ?xxx) (agent_holds ?xxx) (on ?xxx ?xxx) ) )
        (hold (in_motion ?xxx ?xxx) )
        (hold (and (and (and (touch ?xxx) (exists (?o - ball) (touch ?o ?o) ) ) (exists (?k - ball ?p - shelf) (and (touch pink ?p) (not (and (in_motion ?p) (agent_holds ?p agent) ) ) ) ) ) (or (>= 4 (distance ?xxx desk)) (agent_holds ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 9)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (and (not (touch agent ?h) ) (not (agent_holds ?h) ) ) )
          (once (not (object_orientation ?h upright) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (not (and (not (and (and (and (and (agent_holds ?h) (and (and (in_motion ?h sideways) (in_motion ?h ?h) (on bottom_shelf ?h) ) (in ?h) (agent_holds ?h) ) ) (not (> (distance room_center ?h) 2) ) ) (in_motion ?h ?h) ) (in_motion top_drawer) ) ) (or (on ?h) (agent_holds ?h) ) ) ) )
          (once (and (in ?h) (agent_holds ?h) ) )
          (hold-while (on ?h) (and (and (and (in_motion ?h desk) (and (and (exists (?r - hexagonal_bin) (in_motion ?h) ) (not (in_motion ?h) ) (not (in agent agent) ) (agent_holds ?h) (and (not (agent_holds ?h) ) (in_motion ?h) ) (and (or (not (agent_holds ?h) ) (in_motion agent ?h) ) (and (in bed) (on ?h ?h ?h) (not (and (agent_holds ?h) (not (in_motion ?h desk) ) (and (> (distance agent desk) (distance 5 ?h)) (and (and (or (or (agent_holds ?h) (and (equal_x_position bed) (and (and (and (in_motion ?h ?h) (rug_color_under ?h) (adjacent bed) (touch blinds) (in_motion ?h ?h) (and (not (or (= 2 0 8) (and (or (in ?h ?h) (agent_holds agent) ) (<= (distance ?h bed ?h) 5) ) ) ) (in_motion ?h ?h) ) ) (adjacent ?h ?h) (between ?h) ) (or (agent_holds ?h ?h) (game_start agent) ) (in_motion ?h) (adjacent ?h) ) ) ) (in_motion ?h) ) (agent_holds ?h) ) (above ?h ?h) ) ) ) ) ) ) (and (not (agent_holds ?h) ) (in_motion ?h desk) ) (agent_holds ?h floor) (and (not (< (distance 10 agent) 1) ) (on ?h) (agent_holds ?h) (and (object_orientation ?h) (agent_holds ?h) ) ) (agent_holds ?h ?h) ) ) ) (not (in_motion ?h ?h) ) ) ) )
          (once (agent_holds ?h ?h) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (< (distance agent room_center) 1) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (and (not (touch agent ?h) ) (not (agent_holds ?h) ) ) )
          (once (not (object_orientation ?h upright) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference preference3
      (at-end
        (not
          (on ?xxx brown)
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (not (in_motion ?b) ) ) )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (and (not (touch agent ?h) ) (not (agent_holds ?h) ) ) )
          (once (not (object_orientation ?h upright) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:dodgeball) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)


(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
      (game-conserved
        (< (distance ?h ?r) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b) )
          (hold-while (and (not (agent_holds ?b) ) (in_motion ?b) ) (touch ?b ?r) )
          (once (and (in ?h ?b) (forall (?a - dodgeball) (agent_holds ?a) ) ) )
        )
      )
    )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (hold (and (not (touch agent ?h) ) (not (agent_holds ?h) ) ) )
          (once (not (object_orientation ?h upright) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once binKnockedOver) 1 )
)
(:scoring
  (count throwToRampToBin)
)
)

