(define (game game-id) (:domain domain-name)
(:setup
  (exists (?l - (either dodgeball yellow))
    (or
      (or
        (game-conserved
          (in ?l pink)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - dodgeball)
      (and
        (preference preferenceA
          (then
            (once (touch ?d ?d) )
            (hold (in_motion bed) )
            (once (not (on agent) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (> 10 1 )
)
(:scoring
  (+ (count-once-per-objects preferenceA:pink_dodgeball:white) (count-once-per-external-objects preferenceA:dodgeball) )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (in_motion door agent)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?i - doggie_bed)
        (exists (?v - dodgeball)
          (exists (?q - hexagonal_bin)
            (then
              (once (and (<= 1 (distance agent bed)) (in brown) (in ?v) ) )
              (once (on ?v ?q) )
              (once (agent_holds ?q) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (at-end
        (in_motion ?xxx)
      )
    )
  )
)
(:terminal
  (>= (- 10 )
    (count-once-per-objects preferenceA:basketball)
  )
)
(:scoring
  (count-nonoverlapping preferenceA:beachball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?l - flat_block)
    (and
      (and
        (and
          (game-conserved
            (not
              (in_motion ?l)
            )
          )
        )
      )
      (forall (?w - shelf)
        (and
          (and
            (forall (?e - cylindrical_block)
              (game-conserved
                (agent_holds ?e)
              )
            )
          )
        )
      )
      (forall (?q - teddy_bear)
        (and
          (and
            (game-conserved
              (not
                (in_motion ?q)
              )
            )
            (forall (?a ?j ?s - ball)
              (exists (?e - sliding_door)
                (and
                  (exists (?m - (either laptop pillow) ?d - bridge_block)
                    (and
                      (exists (?u - building)
                        (and
                          (game-conserved
                            (agent_holds agent)
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - block)
      (and
        (preference preferenceA
          (exists (?m - chair)
            (then
              (hold (not (in ?q agent) ) )
              (once (in_motion ?q ?q) )
              (once (agent_holds ?m) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (and
      (or
        (and
          (<= 10 (count-nonoverlapping preferenceA:yellow_cube_block) )
        )
        (or
          (> (count-shortest preferenceA:blue_dodgeball) (* (count-once-per-external-objects preferenceA:blue_dodgeball) (* (count-nonoverlapping preferenceA:green) (count-nonoverlapping preferenceA:hexagonal_bin) )
              (+ (* 1 2 )
                5
                (count-nonoverlapping preferenceA:yellow_cube_block:yellow)
              )
            )
          )
        )
      )
    )
    (= (external-forall-maximize (count-once-per-objects preferenceA:hexagonal_bin) ) 10 )
  )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-conserved
      (in_motion ?xxx ?xxx)
    )
    (and
      (game-optional
        (not
          (or
            (and
              (in_motion bed ?xxx)
              (not
                (and
                  (exists (?m - game_object ?g - ball)
                    (and
                      (or
                        (in ?g ?g)
                        (agent_holds agent)
                      )
                      (<= (distance ?g bed ?g) 5)
                    )
                  )
                  (in_motion ?xxx ?xxx)
                )
              )
            )
            (adjacent ?xxx ?xxx)
          )
        )
      )
      (game-optional
        (not
          (in_motion bed ?xxx)
        )
      )
      (exists (?k - hexagonal_bin)
        (game-conserved
          (not
            (open ?k ?k)
          )
        )
      )
    )
    (and
      (game-optional
        (and
          (not
            (touch ?xxx)
          )
          (adjacent )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?d - pillow)
        (at-end
          (and
            (is_setup_object ?d)
            (in_motion ?d)
            (on upside_down)
          )
        )
      )
    )
    (forall (?w - dodgeball)
      (and
        (preference preferenceB
          (then
            (once (in_motion ?w ?w) )
            (once (and (not (agent_holds agent) ) (not (agent_holds ?w ?w) ) (in agent) ) )
            (hold (and (in ?w) (agent_holds ?w) (agent_holds ?w) (= (distance bed ?w)) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ 9 (* (- (>= 10 30 )
        )
        (< (count-nonoverlapping preferenceA:pink_dodgeball:purple) 4 )
      )
    )
    30
  )
)
(:scoring
  8
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (>= (distance_side ?xxx ?xxx) (distance ?xxx ?xxx))
  )
)
(:constraints
  (and
    (forall (?h - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?o - (either golfball rug tall_cylindrical_block))
            (exists (?p - hexagonal_bin ?a - hexagonal_bin)
              (then
                (once (in_motion ?o) )
                (once (and (not (not (and (and (adjacent ?a front) (not (not (in_motion ?h) ) ) (agent_holds ?h) ) (touch ?a pink_dodgeball) ) ) ) (in_motion ?o ?o) (agent_holds floor) ) )
                (once (and (on ?a ?a) (and (in ?o) (not (and (in_motion ?o ?o ?h) (on right ?h) (and (and (not (agent_holds ?a) ) (not (adjacent ?h) ) ) (on ?a) (not (agent_holds ?a ?h) ) (in ?a) ) (and (= 6 4 1) (and (equal_z_position ?a ?o) (not (and (in ?a ?h) (touch ?h) (in desk ?a) (game_over ?a ?o) ) ) ) ) (touch ?a) (and (agent_holds ?a) (in_motion ?o) ) (agent_holds ?a) ) ) (and (on ?h ?o) (agent_holds ?h) ) (< 1 (distance 1 ?h)) ) ) )
              )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?g - dodgeball)
        (then
          (once (and (agent_holds ?g rug) (touch ?g) ) )
          (once (agent_holds ?g) )
          (hold (rug_color_under bed right) )
        )
      )
    )
  )
)
(:terminal
  (>= 4 20 )
)
(:scoring
  (+ 1 (count-longest preferenceB:bed) (<= (* (count-once-per-objects preferenceB:golfball:golfball) (+ 5 (count-nonoverlapping preferenceB:yellow) )
        (count-nonoverlapping preferenceA:beachball)
      )
      (* (+ (+ (count-nonoverlapping preferenceA:yellow) 4 )
        )
        3
        (count-nonoverlapping preferenceB:pink)
        (* 4 )
        (* (count-total preferenceA:tall_cylindrical_block) (- (count-nonoverlapping-measure preferenceA:cube_block) )
        )
        (count-nonoverlapping preferenceB:red:dodgeball:pink)
      )
    )
    (* (count-once-per-objects preferenceA:block) 3 )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?u - red_dodgeball)
    (game-conserved
      (on ?u agent)
    )
  )
)
(:constraints
  (and
    (forall (?b ?q ?i - (either cylindrical_block cube_block key_chain))
      (and
        (preference preferenceA
          (then
            (once (and (and (exists (?p - cube_block ?y - hexagonal_bin) (not (adjacent agent ?b) ) ) (and (adjacent ?i) (touch ?b) ) ) (in_motion agent ?q) ) )
            (once (in_motion ?q) )
            (hold (in ?b) )
          )
        )
        (preference preferenceB
          (exists (?t - teddy_bear)
            (exists (?p - hexagonal_bin)
              (exists (?e - red_dodgeball)
                (exists (?f - shelf ?m - building)
                  (exists (?d - (either blue_cube_block golfball) ?o - (either yellow_cube_block))
                    (exists (?v - teddy_bear)
                      (then
                        (once (agent_holds ?e) )
                        (once (< 0.5 1) )
                        (hold-for 10 (in_motion bed ?q) )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< (* (> 6 3 )
      (count-once-per-objects preferenceA:beachball:yellow_cube_block)
    )
    (total-score)
  )
)
(:scoring
  (count-increasing-measure preferenceB:orange:basketball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (not
    (game-conserved
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?a - cube_block)
        (then
          (once (and (agent_holds ?a floor) (not (equal_z_position ?a) ) ) )
          (once (and (and (and (not (in_motion bed ?a) ) (agent_holds top_shelf) ) (and (not (touch ?a front_left_corner) ) (<= (distance agent room_center agent) 2) ) ) (in_motion ?a) ) )
          (hold (and (< (distance ?a 4) 1) (on ?a) (in ?a) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (and (exists (?u - beachball) (agent_holds ?u) ) (in_motion rug ?xxx) (<= 10 (distance_side 9 agent)) (not (in sideways ?xxx) ) ) )
        (hold (and (agent_holds ?xxx) (< (distance 5 ?xxx) 0.5) ) )
        (once (touch ?xxx) )
      )
    )
    (forall (?a - ball)
      (and
        (preference preferenceC
          (exists (?g - (either laptop hexagonal_bin))
            (exists (?y - shelf)
              (then
                (once (exists (?p - (either cellphone blue_cube_block dodgeball) ?o - curved_wooden_ramp) (in_motion ?o) ) )
                (once (< (distance ?a ?y) 0.5) )
              )
            )
          )
        )
        (preference preferenceD
          (exists (?v - pillow)
            (then
              (hold-while (and (not (agent_holds ?v) ) (not (agent_holds ?a agent) ) ) (not (agent_holds ?a ?a) ) )
              (hold (not (not (not (on tan) ) ) ) )
              (once (and (in agent ?v) (on ?a) (not (= (distance desk 9 room_center) 4) ) ) )
            )
          )
        )
        (preference preferenceE
          (exists (?c ?p ?g ?q ?d ?m - (either doggie_bed alarm_clock))
            (exists (?o - curved_wooden_ramp)
              (then
                (once (= (distance ?c ?q) 1 (distance ?g 10)) )
                (once (agent_holds ?g ?p) )
                (once (adjacent ?g) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-nonoverlapping preferenceA:green) (count-once-per-objects preferenceA:dodgeball) )
)
(:scoring
  (+ (count-nonoverlapping preferenceD:beachball) (count-once-per-objects preferenceE:red) )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (at-end
        (not
          (and
            (not
              (not
                (not
                  (not
                    (and
                      (not
                        (above ?xxx)
                      )
                      (agent_holds ?xxx)
                      (in_motion agent)
                    )
                  )
                )
              )
            )
            (not
              (not
                (and
                  (and
                    (agent_holds door ?xxx)
                    (in_motion ?xxx ?xxx ?xxx)
                  )
                  (adjacent ?xxx top_drawer ?xxx)
                  (and
                    (and
                      (agent_holds ?xxx ?xxx)
                      (not
                        (agent_holds ?xxx)
                      )
                      (exists (?w - (either pen doggie_bed doggie_bed dodgeball))
                        (on ?w ?w)
                      )
                    )
                    (agent_holds ?xxx ?xxx)
                  )
                  (not
                    (in_motion ?xxx)
                  )
                  (agent_holds ?xxx ?xxx)
                  (in_motion agent rug)
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-nonoverlapping preferenceA:golfball:yellow:doggie_bed) (count-once preferenceA:yellow) )
    (>= 2 10 )
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (forall (?v - hexagonal_bin)
      (game-conserved
        (in ?v ?v)
      )
    )
    (forall (?u ?l - hexagonal_bin)
      (exists (?g - hexagonal_bin)
        (and
          (forall (?c - ball)
            (game-optional
              (in_motion ?u)
            )
          )
          (exists (?d - chair)
            (exists (?v - building)
              (exists (?p - (either cd beachball book dodgeball))
                (game-conserved
                  (in ?l desk)
                )
              )
            )
          )
        )
      )
    )
    (forall (?p - (either cylindrical_block dodgeball))
      (and
        (game-conserved
          (on ?p)
        )
        (game-conserved
          (touch ?p ?p)
        )
        (and
          (and
            (game-conserved
              (is_setup_object ?p ?p)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?n - game_object ?u - hexagonal_bin)
        (exists (?v - (either teddy_bear mug dodgeball))
          (exists (?c - hexagonal_bin ?j - (either dodgeball blue_cube_block pyramid_block yellow))
            (then
              (once (same_color ?j ?j) )
              (once (and (in ?v) (adjacent ?j ?v) (and (and (on ?v ?u) (and (and (and (same_color ?j) (agent_holds ?j) ) (in_motion ?j) ) (and (not (same_color green_golfball ?u) ) (exists (?b - (either triangular_ramp)) (on ?v) ) ) ) ) (and (in_motion ?j) (not (in ?j ?v) ) ) ) (and (or (and (agent_holds bed ?v) (in_motion ?u) ) (not (in_motion ?u ?v) ) ) (agent_holds ?v) ) ) )
              (once (in_motion rug ?v) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?l - hexagonal_bin)
        (exists (?k ?z ?m ?r - shelf ?m - dodgeball ?b - hexagonal_bin)
          (then
            (hold-while (not (on ?b) ) (in_motion ?b) )
            (once (and (agent_holds ?b) (exists (?o - (either mug hexagonal_bin) ?y - curved_wooden_ramp) (and (agent_holds ?y ?b) (not (and (in rug) (not (adjacent_side ?b ?b) ) (and (exists (?u - (either yellow_cube_block golfball)) (not (on desk bed) ) ) (adjacent agent) ) (adjacent ?y ?l) ) ) ) ) (agent_holds ?b ?l) (agent_holds ?l ?l) (and (not (on agent ?l) ) (not (in_motion ?b agent) ) ) (same_type ?b ?l) (not (on ?l) ) ) )
            (once (in_motion ?l ?l) )
          )
        )
      )
    )
    (forall (?r - hexagonal_bin ?k - ball ?o - hexagonal_bin)
      (and
        (preference preferenceC
          (exists (?t - cube_block ?b - hexagonal_bin ?d - (either book pyramid_block) ?n - cylindrical_block)
            (then
              (hold (agent_holds ?o) )
              (forall-sequence (?e - cylindrical_block ?y - curved_wooden_ramp)
                (then
                  (hold-while (in ?o agent) (on ?o ?n) )
                  (once (and (agent_holds ?n) (not (agent_holds agent) ) ) )
                  (once (exists (?m - dodgeball) (on desk) ) )
                )
              )
              (hold-while (and (not (and (on ?n) (and (not (or (not (not (on ?n) ) ) (and (and (agent_holds ?n ?n) (and (or (= 0.5 1) (>= 1 4) ) (in ?o) (and (and (agent_holds ?o agent) (agent_holds ?o ?n) ) (not (in ?o) ) ) ) ) (same_color ?o) ) ) ) (not (on ?n) ) ) ) ) (in_motion ?o ?o) (and (in rug) (in_motion ?o) ) (in ?o ?o) (in_motion ?n) (and (and (not (> 1 2) ) (and (on ?n) (not (and (in_motion ?n) (agent_holds ?o) ) ) ) ) (agent_crouches ?n ?n) ) ) (and (agent_holds ?o) (on ?o) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- (total-score) )
    (count-once-per-objects preferenceC:dodgeball)
  )
)
(:scoring
  (count-nonoverlapping preferenceA:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (in ?xxx desk) )
        (once (not (agent_holds ?xxx ?xxx) ) )
        (once (adjacent ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (count-nonoverlapping preferenceA:pink) 10 (* (count-once-per-objects preferenceA:yellow) 0 )
      )
      (count-nonoverlapping preferenceA:doggie_bed)
    )
    (>= (count-nonoverlapping preferenceA:yellow) 4 )
    (or
      (>= 25 (count-same-positions preferenceA:blue_dodgeball) )
      (> 2 (+ 4 2 )
      )
    )
    (>= (* (count-nonoverlapping preferenceA:basketball:book) (count-once preferenceA:green) )
      (<= (count-once preferenceA:pink_dodgeball:golfball) (count-nonoverlapping preferenceA:beachball:pink) )
    )
  )
)
(:scoring
  (count-increasing-measure preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (forall (?u - dodgeball ?b - game_object)
        (and
          (forall (?d - beachball ?v - hexagonal_bin)
            (forall (?x - block)
              (and
                (forall (?c - (either yellow pillow))
                  (exists (?u - hexagonal_bin)
                    (and
                      (game-optional
                        (and
                          (touch ?c)
                          (not
                            (and
                              (= (distance ?u ?c 10) (distance ))
                              (not
                                (not
                                  (agent_holds ?x)
                                )
                              )
                              (not
                                (adjacent_side ?x ?u)
                              )
                            )
                          )
                        )
                      )
                      (game-optional
                        (not
                          (and
                            (in_motion ?x)
                            (on ?b)
                          )
                        )
                      )
                      (exists (?t - pyramid_block)
                        (or
                          (and
                            (exists (?w - book ?a - dodgeball)
                              (game-conserved
                                (not
                                  (and
                                    (on ?u)
                                    (and
                                      (in_motion ?v ?b ?a)
                                      (and
                                        (< 1 4)
                                        (and
                                          (not
                                            (and
                                              (in ?t)
                                              (in ?t)
                                              (not
                                                (and
                                                  (agent_holds ?b)
                                                  (and
                                                    (and
                                                      (game_over ?a)
                                                      (not
                                                        (in_motion ?b ?t)
                                                      )
                                                    )
                                                    (= 1 (distance 7 7))
                                                  )
                                                  (adjacent ?c)
                                                )
                                              )
                                            )
                                          )
                                          (not
                                            (in_motion ?v rug)
                                          )
                                        )
                                      )
                                    )
                                    (and
                                      (in_motion ?v)
                                      (not
                                        (< 1 (distance agent ?a))
                                      )
                                    )
                                  )
                                )
                              )
                            )
                          )
                          (or
                            (game-conserved
                              (in_motion ?u)
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (and
            (exists (?c - red_dodgeball)
              (forall (?r - hexagonal_bin)
                (exists (?m - block)
                  (and
                    (exists (?v ?w - wall)
                      (and
                        (game-conserved
                          (and
                            (agent_holds ?m)
                            (= (distance_side ?c agent))
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
            (and
              (exists (?z - dodgeball)
                (and
                  (forall (?v - hexagonal_bin)
                    (game-conserved
                      (in_motion ?z ?b)
                    )
                  )
                )
              )
              (game-conserved
                (and
                  (agent_holds ?b agent)
                  (agent_holds ?b front)
                )
              )
            )
            (game-conserved
              (and
                (or
                  (and
                    (and
                      (agent_holds ?b)
                    )
                    (not
                      (touch ?b)
                    )
                  )
                  (forall (?r - red_dodgeball)
                    (adjacent ?b ?r)
                  )
                )
                (and
                  (in ?b ?b)
                  (not
                    (agent_holds ?b ?b)
                  )
                  (is_setup_object ?b ?b)
                )
                (agent_holds ?b ?b)
                (object_orientation ?b)
              )
            )
          )
          (exists (?j ?u - (either dodgeball hexagonal_bin rug))
            (exists (?f - dodgeball)
              (and
                (exists (?s - dodgeball ?q - hexagonal_bin)
                  (game-optional
                    (agent_holds agent)
                  )
                )
              )
            )
          )
        )
      )
      (game-conserved
        (not
          (touch ?xxx ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (at-end
        (agent_holds right ?xxx)
      )
    )
    (preference preferenceB
      (then
        (once (in_motion ?xxx) )
        (hold (in_motion ?xxx) )
        (any)
      )
    )
  )
)
(:terminal
  (>= (* (count-nonoverlapping preferenceA:basketball:basketball) )
    (total-time)
  )
)
(:scoring
  (+ (count-once preferenceB:purple) 2 )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?r - (either golfball lamp wall dodgeball key_chain cd cylindrical_block))
    (exists (?f ?q - (either laptop golfball))
      (game-conserved
        (or
          (not
            (and
              (not
                (< 10 6)
              )
              (exists (?o ?c - game_object)
                (agent_holds ?o ?r)
              )
            )
          )
          (agent_holds ?q)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (any)
        (once (agent_holds front ?xxx) )
        (hold-while (equal_z_position ?xxx ?xxx) (in_motion ?xxx ?xxx) (and (and (object_orientation ?xxx ?xxx) (not (not (not (not (and (and (in_motion ?xxx ?xxx) (and (and (agent_holds ?xxx ?xxx) (or (not (on sideways ?xxx) ) (in ?xxx) ) ) (= 9 2 7) ) ) (not (not (agent_holds ?xxx) ) ) ) ) ) ) ) ) (in_motion agent) (same_color ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (+ (+ (or 10 (+ 4 )
          (+ (* (count-once-per-objects preferenceA:doggie_bed) 8 )
            (* 2 (count-nonoverlapping preferenceA:basketball) (count-once-per-objects preferenceA:dodgeball:book) )
          )
        )
        50
        4
        (* (total-score) 3 )
        (count-nonoverlapping preferenceA:tall_cylindrical_block)
        (= (+ (- (+ 7 (count-nonoverlapping preferenceA:yellow) )
            )
            (count-once preferenceA:golfball)
          )
          5
        )
      )
      (count-nonoverlapping preferenceA:golfball:hexagonal_bin)
    )
    (* (+ (* 9 (count-nonoverlapping preferenceA:red) )
        (count-nonoverlapping preferenceA:golfball)
      )
    )
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (exists (?i - hexagonal_bin)
        (game-optional
          (and
            (touch ?i)
            (not
              (agent_holds ?i)
            )
          )
        )
      )
      (game-conserved
        (not
          (in_motion ?xxx ?xxx)
        )
      )
      (and
        (and
          (game-conserved
            (and
              (same_color ?xxx)
              (agent_holds rug ?xxx)
            )
          )
          (game-conserved
            (in_motion ?xxx)
          )
          (forall (?w - dodgeball)
            (forall (?e - (either doggie_bed doggie_bed) ?x - hexagonal_bin)
              (forall (?p - wall)
                (exists (?l - ball)
                  (exists (?j - hexagonal_bin ?z - hexagonal_bin ?d - chair)
                    (game-optional
                      (in ?d ?w)
                    )
                  )
                )
              )
            )
          )
          (exists (?b ?r - block ?p - hexagonal_bin)
            (and
              (exists (?u - cube_block)
                (game-optional
                  (touch floor)
                )
              )
            )
          )
          (forall (?g - (either blue_cube_block pyramid_block) ?t - hexagonal_bin)
            (not
              (forall (?w - (either tall_cylindrical_block laptop) ?x - flat_block)
                (and
                  (and
                    (exists (?z - shelf)
                      (game-conserved
                        (in ?x rug floor)
                      )
                    )
                    (game-conserved
                      (in_motion ?t)
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
    (and
      (game-conserved
        (< 8 (distance 2 ?xxx))
      )
      (and
        (exists (?x ?v - cube_block)
          (exists (?s - drawer)
            (and
              (game-conserved
                (in ?v ?x)
              )
              (and
                (and
                  (forall (?b - hexagonal_bin)
                    (and
                      (forall (?q - (either bed cd) ?d - ball ?f ?e - pyramid_block)
                        (forall (?g - (either cube_block dodgeball hexagonal_bin watch))
                          (exists (?c - hexagonal_bin)
                            (game-optional
                              (adjacent ?s ?s)
                            )
                          )
                        )
                      )
                    )
                  )
                  (exists (?w - hexagonal_bin)
                    (and
                      (and
                        (exists (?g - triangular_ramp ?u ?m - wall ?g - hexagonal_bin)
                          (exists (?l - hexagonal_bin)
                            (game-conserved
                              (in_motion ?s ?x)
                            )
                          )
                        )
                        (game-optional
                          (in_motion east_sliding_door desk)
                        )
                      )
                    )
                  )
                )
              )
              (exists (?i - game_object)
                (and
                  (and
                    (game-optional
                      (and
                        (not
                          (and
                            (in_motion bed)
                            (agent_holds ?s)
                          )
                        )
                        (and
                          (not
                            (not
                              (in ?i bed)
                            )
                          )
                          (in_motion bed ?i)
                        )
                      )
                    )
                    (and
                      (game-optional
                        (and
                          (on ?v)
                          (not
                            (not
                              (not
                                (agent_holds ?s)
                              )
                            )
                          )
                        )
                      )
                      (exists (?q - chair)
                        (game-optional
                          (not
                            (adjacent ?i)
                          )
                        )
                      )
                    )
                  )
                  (not
                    (game-optional
                      (not
                        (touch ?i ?s)
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?v ?w - bridge_block)
        (then
          (once (in_motion agent ?w) )
          (once (and (not (not (in ?w) ) ) (agent_holds ?v ?w) ) )
          (once (forall (?b - (either pyramid_block red)) (in_motion ?v ?b) ) )
        )
      )
    )
    (forall (?t - cube_block ?r - color ?h - red_pyramid_block)
      (and
        (preference preferenceB
          (exists (?n - dodgeball ?x - (either cd dodgeball golfball laptop) ?z - game_object ?i - doggie_bed)
            (then
              (once (in agent ?h) )
              (once (and (in ?i) (in_motion ?i) (exists (?y - cube_block) (on desk) ) ) )
              (once (in_motion ?h ?h) )
            )
          )
        )
        (preference preferenceC
          (then
            (hold (or (and (rug_color_under ?h ?h) (in ?h) (and (and (agent_holds ?h) ) (in ?h) ) (and (not (in_motion ?h) ) (in_motion ?h ?h) ) ) (object_orientation ?h ?h) (same_color ?h front) ) )
            (once (in_motion ?h ?h) )
            (once (not (and (agent_holds upright) (on ?h ?h) ) ) )
          )
        )
        (preference preferenceD
          (at-end
            (object_orientation green)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (count-nonoverlapping preferenceA:orange:basketball) )
)
(:scoring
  (- 5 )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (game_over ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once-measure (agent_holds ?xxx) (distance ?xxx room_center) )
        (once (same_object ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count-nonoverlapping preferenceA:yellow:pink) (external-forall-minimize (count-once-per-objects preferenceA:yellow_pyramid_block) ) )
)
(:scoring
  (count-nonoverlapping preferenceA:beachball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-conserved
      (on ?xxx)
    )
    (exists (?c - building ?u - (either pyramid_block book))
      (game-optional
        (not
          (agent_holds ?u)
        )
      )
    )
    (game-conserved
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?k - ball ?e - tall_cylindrical_block)
      (and
        (preference preferenceA
          (then
            (once (not (agent_holds ?e) ) )
            (hold (touch ?e ?e) )
            (once (in ?e ?e) )
            (once (and (exists (?n ?d - hexagonal_bin) (in_motion bed desk) ) (open ?e) ) )
          )
        )
      )
    )
    (forall (?b - dodgeball)
      (and
        (preference preferenceB
          (then
            (hold (exists (?c - red_pyramid_block) (not (in_motion ?b) ) ) )
            (once-measure (on ?b ?b) (distance ?b agent) )
            (once (not (not (and (in_motion ?b brown) (in ?b) ) ) ) )
          )
        )
      )
    )
    (preference preferenceC
      (then
        (once (in ?xxx ?xxx) )
        (once (in agent) )
        (once (< (distance ?xxx bed) (distance ?xxx ?xxx)) )
      )
    )
  )
)
(:terminal
  (>= (count-total preferenceB:orange) (count-nonoverlapping preferenceA:basketball) )
)
(:scoring
  (count-nonoverlapping preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (exists (?o - hexagonal_bin)
      (on ?o ?o)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold (on agent) )
        (once (and (agent_holds ?xxx ?xxx) (< 8 1) ) )
        (once (and (adjacent ?xxx ?xxx) (agent_holds ?xxx ?xxx) (touch agent) (in ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (> (count-nonoverlapping preferenceA:green) 1 )
)
(:scoring
  3
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?r - block ?o - building)
    (game-optional
      (and
        (not
          (< 1 (distance 5 room_center))
        )
        (in_motion agent ?o)
      )
    )
  )
)
(:constraints
  (and
    (forall (?y - dodgeball)
      (and
        (preference preferenceA
          (exists (?m ?o ?q ?x ?l ?f - chair)
            (at-end
              (agent_holds ?l)
            )
          )
        )
        (preference preferenceB
          (then
            (once (agent_holds ?y agent) )
            (once (in_motion ?y ?y) )
            (once (in_motion ?y) )
          )
        )
        (preference preferenceC
          (exists (?v - block)
            (at-end
              (adjacent_side ?y ?y)
            )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?i - (either dodgeball dodgeball))
        (then
          (once (agent_holds ?i ?i) )
          (once (agent_holds ?i) )
          (any)
        )
      )
    )
    (preference preferenceE
      (then
        (hold (not (agent_holds ?xxx) ) )
        (once (game_over ?xxx ?xxx) )
        (once (object_orientation ?xxx) )
      )
    )
    (forall (?f - hexagonal_bin)
      (and
        (preference preferenceF
          (exists (?m - cube_block)
            (exists (?n ?k ?p - cube_block ?o ?g - (either cylindrical_block hexagonal_bin laptop))
              (then
                (once (in_motion ?f) )
                (once (in_motion ?g) )
                (once (on ?f ?o) )
              )
            )
          )
        )
      )
    )
    (preference preferenceG
      (exists (?a - hexagonal_bin)
        (then
          (once (not (not (not (on agent) ) ) ) )
          (once (or (not (and (in_motion ?a) (in_motion ?a ?a) (exists (?v - curved_wooden_ramp) (or (adjacent ?a) (in_motion bed ?v) ) ) ) ) (exists (?z - game_object) (in_motion ?a ?z ?a) ) ) )
          (once (forall (?n - chair) (on ?n) ) )
          (once (in ?a ?a) )
        )
      )
    )
    (forall (?y - dodgeball ?r - beachball)
      (and
        (preference preferenceH
          (then
            (hold (or (same_object agent ?r) (in ?r) ) )
            (hold (agent_holds desk) )
            (hold (= (building_size ) 0.5 (distance ?r agent)) )
          )
        )
        (preference preferenceI
          (exists (?p - ball)
            (then
              (once (in_motion ?p) )
              (hold (in_motion pink ?r) )
              (once (and (agent_holds ?r ?p) (not (and (> 1 3) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count-nonoverlapping preferenceA:doggie_bed) 5 )
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (is_setup_object ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?f - hexagonal_bin)
        (then
          (hold (not (and (and (and (forall (?h - hexagonal_bin) (and (not (exists (?a - hexagonal_bin) (agent_holds ?f ?f) ) ) (in ?h ?f) (exists (?y - doggie_bed ?b - game_object) (not (object_orientation agent) ) ) (touch ?f) (agent_holds ?h ?h) (and (in ?f) (in upright floor ?f ?f) ) ) ) (rug_color_under ?f) ) (not (equal_z_position agent) ) ) (agent_holds ?f ?f) ) ) )
          (once (= 2 1) )
          (hold (agent_holds ?f ?f) )
        )
      )
    )
  )
)
(:terminal
  (> (- 1 )
    (count-nonoverlapping preferenceA:dodgeball)
  )
)
(:scoring
  2
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (not
      (not
        (and
          (not
            (not
              (< (distance ?xxx ?xxx) (distance side_table ?xxx))
            )
          )
          (not
            (and
              (and
                (and
                  (in_motion ?xxx)
                  (agent_holds agent)
                )
                (agent_holds ?xxx ?xxx)
              )
              (on ?xxx)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - block)
      (and
        (preference preferenceA
          (then
            (once (agent_holds rug) )
            (once (not (agent_holds ?q) ) )
            (once (in_motion agent yellow) )
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (same_color ?xxx ?xxx) )
        (once (on ?xxx) )
        (once (agent_holds floor) )
      )
    )
  )
)
(:terminal
  (> 2 4 )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (exists (?p - block)
      (exists (?z - dodgeball ?y - hexagonal_bin ?w - ball)
        (game-optional
          (not
            (type ?p)
          )
        )
      )
    )
    (exists (?h - (either teddy_bear pen))
      (game-conserved
        (and
          (not
            (touch ?h ?h)
          )
          (same_color ?h)
        )
      )
    )
    (game-conserved
      (exists (?v - hexagonal_bin)
        (not
          (agent_holds ?v ?v)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?x - color)
        (exists (?i - block ?w - dodgeball)
          (exists (?g - dodgeball)
            (exists (?j - cube_block)
              (then
                (once (and (in_motion ?j) (not (adjacent ?g) ) (rug_color_under ?w) ) )
                (once (agent_holds rug ?w) )
                (once (in_motion ?w) )
              )
            )
          )
        )
      )
    )
    (forall (?p ?j - (either key_chain golfball tall_cylindrical_block teddy_bear curved_wooden_ramp cube_block main_light_switch))
      (and
        (preference preferenceB
          (exists (?e - dodgeball)
            (then
              (hold (in_motion ?j agent) )
              (once (agent_holds rug) )
              (hold-while (on ?e ?p) (in_motion ?e) )
            )
          )
        )
      )
    )
    (forall (?t - red_dodgeball ?v - wall ?q - hexagonal_bin)
      (and
        (preference preferenceC
          (exists (?s - game_object)
            (then
              (once (and (agent_holds door floor) (in_motion ?q ?s) ) )
              (once (in_motion blue) )
              (once (same_color ?s ?q pillow ?s) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (* (count-nonoverlapping preferenceB:pink:green:orange) (count-once-per-objects preferenceC:yellow_cube_block) )
      50
    )
    (>= 2 (total-time) )
    (< 30 (count-nonoverlapping preferenceC:dodgeball) )
  )
)
(:scoring
  (* (* (- 2 )
    )
    (* (* (count-once preferenceA:golfball) (count-once-per-objects preferenceA:blue_dodgeball:pink) )
      (count-unique-positions preferenceB:dodgeball)
      10
      (count-nonoverlapping preferenceB:green:golfball:red_pyramid_block)
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?b - cube_block)
    (game-optional
      (not
        (< 1 (distance desk room_center))
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (in_motion agent) )
        (once-measure (not (agent_holds ?xxx) ) (distance room_center desk) )
        (once (in_motion floor) )
      )
    )
  )
)
(:terminal
  (> (= (/ 7 (total-score) ) 3 )
    (not
      8
    )
  )
)
(:scoring
  (count-once preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?r - color)
    (game-conserved
      (in_motion ?r top_shelf)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?w - shelf)
        (exists (?b - (either dodgeball triangle_block pillow) ?f - triangular_ramp)
          (exists (?c - building)
            (then
              (once (adjacent ?f) )
              (once (not (and (not (not (adjacent ?f) ) ) (adjacent pillow) ) ) )
              (hold (not (and (= (distance ?f) (distance ?c ?f)) (agent_holds ?f) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (external-forall-maximize 4 ) (count-nonoverlapping preferenceA:pink:dodgeball) )
      6
    )
  )
)
(:scoring
  (* (count-nonoverlapping preferenceA:golfball:doggie_bed) )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (on ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?j - cube_block)
        (then
          (hold (adjacent ?j top_drawer ?j) )
          (hold (in_motion ?j ?j) )
          (once (in_motion ?j) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (and (on door) (agent_holds ?xxx) ) )
        (once (in ?xxx ?xxx) )
        (once (exists (?w - (either pyramid_block alarm_clock)) (rug_color_under ?w) ) )
      )
    )
  )
)
(:terminal
  (< 7 5 )
)
(:scoring
  (total-score)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (in_motion green_golfball)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?y - hexagonal_bin)
        (then
          (once (not (on ?y) ) )
          (once (in_motion desktop ?y ?y) )
          (once (agent_holds ?y ?y) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (adjacent_side ?xxx left) )
        (once (adjacent ?xxx) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
    (forall (?b - golfball)
      (and
        (preference preferenceC
          (exists (?w - cube_block)
            (exists (?x - golfball ?g - dodgeball)
              (exists (?x - (either pyramid_block pencil))
                (then
                  (once (or (in bed) (in_motion ?g) ) )
                  (hold (same_color ?g) )
                  (once (and (not (not (adjacent rug) ) ) (and (and (in ?w) (not (and (exists (?z - dodgeball) (in_motion ?w ?z) ) (in_motion ?x) ) ) (in_motion ?x west_wall) ) (adjacent ?x ?g) (on ?w) ) (not (in_motion ?g ?g) ) (not (touch ?x ?x) ) (in ?w ?w) (not (in_motion desk) ) (agent_holds door ?g) (agent_holds ?g) ) )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (count-once-per-objects preferenceA:purple) (* (count-once-per-objects preferenceC:cylindrical_block) 12 3 (count-overlapping preferenceA:basketball) (* (count-nonoverlapping preferenceA:golfball) (* (not 10 ) (count-once-per-objects preferenceC:golfball) )
        )
        (or
          (count-nonoverlapping preferenceA:pink_dodgeball)
          (count-nonoverlapping preferenceC:yellow_cube_block:beachball)
        )
      )
    )
    8
  )
)
(:scoring
  (< 5 (* (count-once preferenceB:red:beachball) (* (count-once preferenceA:pink) (count-once-per-external-objects preferenceB:pink) )
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?y - game_object)
    (forall (?o - (either alarm_clock cylindrical_block))
      (game-optional
        (and
          (in_motion ?o)
          (agent_holds ?y)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (at-end
        (not
          (and
            (on ?xxx)
            (and
              (adjacent ?xxx ?xxx)
              (and
                (on ?xxx)
                (not
                  (in_motion ?xxx)
                )
              )
            )
          )
        )
      )
    )
    (forall (?m - hexagonal_bin ?b - ball)
      (and
        (preference preferenceB
          (exists (?k - dodgeball ?z - hexagonal_bin)
            (then
              (once (not (not (and (agent_holds pink desk) (in ?z ?b) ) ) ) )
              (once (and (in_motion ?b ?z) (agent_holds ?b) ) )
              (once (not (type ?b) ) )
              (once (not (in_motion agent ?z) ) )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?s - pillow)
        (then
          (once (adjacent_side ?s) )
          (once (object_orientation ?s) )
        )
      )
    )
  )
)
(:terminal
  (or
    (= (+ (count-once-per-objects preferenceA:purple) (count-nonoverlapping preferenceC:cube_block:golfball) )
      3
    )
    (>= (and 2 15 8 ) 1 )
    (>= 18 (count-nonoverlapping preferenceC:dodgeball) )
  )
)
(:scoring
  (count-once-per-external-objects preferenceB:basketball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?z - ball)
    (forall (?q ?u - hexagonal_bin)
      (game-conserved
        (forall (?c - game_object)
          (in_motion ?u ?u)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?m - ball)
        (exists (?x - ball)
          (then
            (hold (in_motion ?m) )
            (once (in_motion ?x) )
            (hold (in_motion ?x) )
          )
        )
      )
    )
  )
)
(:terminal
  (= (total-score) 6 )
)
(:scoring
  10
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?c - doggie_bed)
        (at-end
          (and
            (object_orientation back ?c)
            (on ?c)
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (not (exists (?n - cube_block) (< (distance ) 3) ) ) )
        (once (in_motion ?xxx ?xxx) )
        (once (in ?xxx) )
      )
    )
    (preference preferenceC
      (exists (?s - building)
        (at-end
          (not
            (in_motion ?s)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preferenceA:beachball) (count-nonoverlapping preferenceA:pink) )
)
(:scoring
  (total-score)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (and
      (adjacent pink_dodgeball ?xxx)
      (not
        (not
          (adjacent pink)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?c - block)
      (and
        (preference preferenceA
          (then
            (once (< 6 1) )
            (once (and (not (agent_holds ?c ?c) ) (agent_holds ?c ?c) ) )
            (once (exists (?v - building) (on ?c ?v) ) )
          )
        )
        (preference preferenceB
          (then
            (once-measure (in_motion ?c ?c) (distance 2 door) )
            (once (not (type rug) ) )
            (once (same_color ?c ?c) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preferenceA:side_table) (* (count-once-per-objects preferenceB:dodgeball) 10 )
  )
)
(:scoring
  (* (count-once preferenceB:beachball) (* (count-nonoverlapping preferenceB:orange:beachball) (< 7 6 )
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?v ?h ?z - teddy_bear ?s - wall)
    (exists (?n - (either pyramid_block alarm_clock))
      (exists (?v - dodgeball ?o - hexagonal_bin)
        (game-optional
          (and
            (and
              (same_color ?s)
              (< 1 (distance ))
              (exists (?k - game_object)
                (< 1 1)
              )
              (in_motion ?n agent)
            )
            (on ?o right)
            (not
              (agent_holds ?n)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?j - block ?g - drawer ?u - book ?s - hexagonal_bin ?c - ball)
      (and
        (preference preferenceA
          (exists (?w - drawer ?z ?p ?i - triangular_ramp)
            (exists (?g - dodgeball)
              (then
                (hold (and (agent_holds rug) (agent_holds ?g) ) )
                (once (< 2 4) )
                (once (on ?z ?g) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (and
      (>= 5 (* 20 5 )
      )
      (>= 3 (+ 9 (* (count-nonoverlapping preferenceA:yellow:dodgeball) (+ 10 (count-nonoverlapping preferenceA:dodgeball) (* 3 (count-once-per-objects preferenceA:blue_cube_block) )
              10
              9
              8
              (* (+ (* (count-nonoverlapping preferenceA:golfball) (count-once-per-objects preferenceA:blue_pyramid_block) )
                  (count-once preferenceA:basketball:dodgeball)
                )
                180
              )
              (count-nonoverlapping preferenceA:dodgeball)
              (count-nonoverlapping preferenceA:basketball)
              5
              5
              (count-same-positions preferenceA:dodgeball:dodgeball)
            )
          )
        )
      )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-conserved
      (= 8 (distance ?xxx agent desk))
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (in upright) )
        (once (and (in_motion ?xxx) (touch ?xxx) ) )
        (hold (adjacent ?xxx back) )
      )
    )
    (preference preferenceB
      (exists (?s - ball)
        (then
          (hold-while (and (in ?s) (and (and (and (< (distance ?s) (distance ?s ?s)) (on ?s ?s) ) (in ?s ?s) ) (and (in_motion left ?s) (not (between ?s) ) ) (in agent) ) (and (on ?s) (and (adjacent_side ?s ?s) (not (and (and (in desk) (on ?s) (agent_holds ?s) ) (on pillow) (not (on ?s ?s) ) (and (and (exists (?i - dodgeball) (adjacent ?s ?i) ) (agent_holds ?s ?s) ) (and (> 2 (distance room_center door)) (in ?s) ) (and (and (rug_color_under ?s) (adjacent ?s) ) (> (distance ?s 9) 2) ) ) (in_motion pink) (in ?s) (and (adjacent ?s) (and (touch ?s ?s) ) ) ) ) ) ) (faces ?s) ) (not (same_color ?s) ) )
          (once (equal_z_position ?s ?s agent) )
          (once (not (in ?s ?s) ) )
        )
      )
    )
    (preference preferenceC
      (then
        (hold (agent_holds bed) )
        (once (in floor ?xxx) )
        (once (not (not (= 2 2) ) ) )
      )
    )
  )
)
(:terminal
  (not
    (> (* (count-nonoverlapping preferenceC:dodgeball) (external-forall-maximize (count-nonoverlapping preferenceC:beachball) ) )
      (count-once-per-objects preferenceC:beachball:pink_dodgeball)
    )
  )
)
(:scoring
  (count-nonoverlapping preferenceB)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (on front)
  )
)
(:constraints
  (and
    (forall (?w - dodgeball)
      (and
        (preference preferenceA
          (then
            (once (rug_color_under ?w ?w) )
            (once (agent_holds ?w ?w ?w) )
            (once (in_motion ?w) )
          )
        )
        (preference preferenceB
          (then
            (once (and (touch ?w) (agent_holds pink) ) )
            (hold-while (forall (?r - (either cellphone desktop mug dodgeball yellow desktop dodgeball)) (adjacent ?r) ) (in_motion ?w) (on ?w) )
            (once (and (in ?w) (and (on desk floor) (in ?w) ) ) )
          )
        )
      )
    )
    (forall (?h - golfball)
      (and
        (preference preferenceC
          (exists (?t - game_object)
            (exists (?g - doggie_bed)
              (exists (?u - drawer)
                (then
                  (hold (in_motion agent ?u) )
                  (once (and (agent_holds ?u) (on ?t) ) )
                  (hold (agent_holds ?t ?t) )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* 7 5 (count-nonoverlapping preferenceB:blue_dodgeball) (count-once-per-objects preferenceC:dodgeball:basketball) 3 (+ (+ (count-nonoverlapping preferenceA:yellow) (= 5 4 )
        )
        (count-once-per-external-objects preferenceC:basketball)
      )
    )
    (count-unique-positions preferenceA:red)
  )
)
(:scoring
  3
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (on ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?k - dodgeball)
        (then
          (once (agent_holds ?k ?k) )
          (hold (in_motion ?k ?k) )
          (hold-while (agent_holds ?k ?k) (agent_holds ?k) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (agent_holds ?xxx) )
        (hold (agent_holds desk ?xxx) )
        (once (not (in_motion ?xxx) ) )
      )
    )
    (preference preferenceC
      (then
        (once (on ?xxx ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
        (once (agent_holds bed) )
      )
    )
  )
)
(:terminal
  (> 30 (count-nonoverlapping preferenceB:cube_block:basketball:golfball) )
)
(:scoring
  (count-nonoverlapping preferenceA:basketball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?u - drawer)
    (game-conserved
      (and
        (in_motion ?u ?u)
        (in_motion green_golfball)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?q ?h - (either cd golfball blue_cube_block yellow_cube_block))
        (exists (?l - hexagonal_bin)
          (then
            (once (not (and (in desk) (and (above ?l) (not (touch ?q ?l) ) ) ) ) )
            (hold-while (agent_holds ?q) (or (not (in_motion ?l ?h) ) ) (and (exists (?d - block) (not (and (agent_holds ?d ?q) (in_motion ?h ?d) ) ) ) (same_type ?h) (in ?h) ) )
            (once (not (not (and (in ?l yellow) (agent_holds ?l) ) ) ) )
          )
        )
      )
    )
    (forall (?k - teddy_bear ?z - game_object)
      (and
        (preference preferenceB
          (then
            (once (not (adjacent ) ) )
            (hold (in_motion ?z ?z) )
            (hold (not (in_motion floor ?z) ) )
          )
        )
        (preference preferenceC
          (at-end
            (on ?z ?z)
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (and
      (>= (= 3 (count-nonoverlapping preferenceC:pyramid_block) 5 )
        (external-forall-minimize
          (* 4 18 )
        )
      )
      (>= (count-once-per-objects preferenceA:pink:rug) (count-nonoverlapping preferenceB) )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (forall (?b - dodgeball)
      (forall (?o - hexagonal_bin)
        (forall (?e - tall_cylindrical_block)
          (exists (?s - hexagonal_bin)
            (game-conserved
              (and
                (adjacent ?e ?b)
                (on ?b ?b)
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?w - (either blue_cube_block wall))
            (exists (?c - shelf)
              (at-end
                (not
                  (in_motion ?r)
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (and (and (and (in_motion ?xxx ?xxx) (agent_holds ?xxx desk) ) (adjacent ?xxx ?xxx) ) (on ?xxx ?xxx) ) )
        (hold (in ?xxx) )
        (hold (and (not (on bed) ) (and (agent_holds agent ?xxx) (agent_holds ?xxx) ) ) )
      )
    )
    (preference preferenceC
      (at-end
        (touch desk)
      )
    )
  )
)
(:terminal
  (>= 8 (- (count-nonoverlapping preferenceB:cube_block) )
  )
)
(:scoring
  (count-once-per-objects preferenceB:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?j - (either tall_cylindrical_block curved_wooden_ramp) ?w - shelf ?b - teddy_bear)
    (forall (?t - curved_wooden_ramp)
      (game-conserved
        (agent_holds ?b)
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - cube_block)
      (and
        (preference preferenceA
          (exists (?e - blue_pyramid_block ?y - tall_cylindrical_block ?h - building ?i - teddy_bear)
            (exists (?z - dodgeball)
              (then
                (hold-while (and (in_motion ?q ?z) (exists (?d - hexagonal_bin ?g - dodgeball) (and (agent_holds ?g ?z) (and (not (in_motion desktop) ) (not (not (and (in_motion ?g) (touch ?g) ) ) ) ) ) ) ) (and (not (on ?z) ) (not (< 5 1) ) (agent_holds ?q ?z) ) )
                (once (not (agent_holds ?i rug) ) )
                (hold (and (< (distance ?z ?q) (distance ?i desk)) (not (and (adjacent_side side_table ?z) (exists (?y - hexagonal_bin) (adjacent ?z) ) ) ) ) )
              )
            )
          )
        )
      )
    )
    (forall (?l - hexagonal_bin)
      (and
        (preference preferenceB
          (exists (?x - rug)
            (exists (?i - hexagonal_bin ?n - (either tall_cylindrical_block triangle_block))
              (exists (?c ?p - hexagonal_bin)
                (at-end
                  (in_motion ?x)
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ 3 3 )
    3
  )
)
(:scoring
  (count-once-per-external-objects preferenceB:red_pyramid_block)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (or
      (or
        (game-conserved
          (and
            (and
              (in ?xxx ?xxx)
              (in_motion ?xxx ?xxx)
            )
            (and
              (rug_color_under ?xxx)
              (and
                (agent_holds agent)
                (in_motion ?xxx)
                (= 1 1)
              )
              (touch ?xxx ?xxx)
              (<= (distance ?xxx back) (distance agent))
              (not
                (faces ?xxx ?xxx)
              )
              (agent_holds ?xxx side_table)
              (< 3 8)
              (agent_holds ?xxx)
              (and
                (and
                  (same_color agent ?xxx)
                  (< (distance ?xxx) (distance ?xxx))
                )
                (agent_holds ?xxx)
                (in_motion ?xxx ?xxx ?xxx)
              )
              (agent_holds ?xxx)
              (rug_color_under ?xxx)
              (exists (?m - hexagonal_bin)
                (and
                  (and
                    (not
                      (adjacent ?m)
                    )
                    (not
                      (in_motion ?m ?m)
                    )
                    (>= 9 1)
                    (on ?m)
                  )
                  (and
                    (in_motion ?m ?m)
                    (in ?m ?m)
                    (not
                      (between ?m ?m yellow)
                    )
                  )
                )
              )
            )
          )
        )
        (exists (?k ?q - book)
          (game-optional
            (exists (?v - cube_block ?u - red_pyramid_block)
              (and
                (not
                  (in_motion left ?k)
                )
                (agent_holds ?q)
                (and
                  (agent_holds ?u ?k)
                  (or
                    (agent_holds bed)
                    (and
                      (in ?u)
                      (on ?q)
                    )
                    (agent_holds ?u)
                  )
                )
              )
            )
          )
        )
        (forall (?y - (either pencil))
          (exists (?p - hexagonal_bin ?q - cube_block)
            (game-optional
              (in_motion ?q agent)
            )
          )
        )
      )
      (forall (?b - doggie_bed)
        (or
          (game-conserved
            (agent_holds ?b)
          )
          (game-optional
            (adjacent agent ?b)
          )
        )
      )
      (and
        (and
          (exists (?b - (either key_chain beachball yellow_cube_block))
            (and
              (and
                (and
                  (forall (?v - hexagonal_bin)
                    (and
                      (game-optional
                        (and
                          (agent_holds bed ?v)
                          (and
                            (agent_holds ?b)
                            (not
                              (not
                                (not
                                  (and
                                    (forall (?u - ball)
                                      (agent_holds rug)
                                    )
                                    (not
                                      (agent_holds agent)
                                    )
                                  )
                                )
                              )
                            )
                            (in ?b)
                            (on ?b)
                            (on ?b ?v ?v)
                            (in_motion desk)
                            (on ?v blue)
                            (exists (?x - curved_wooden_ramp)
                              (and
                                (and
                                  (agent_holds ?v)
                                  (on ?x ?b)
                                )
                              )
                            )
                            (and
                              (exists (?e - hexagonal_bin)
                                (not
                                  (in_motion ?e ?v)
                                )
                              )
                              (agent_holds pink_dodgeball)
                              (> (distance room_center ?b) (distance room_center 2))
                            )
                            (and
                              (in bed)
                              (in upside_down ?v floor)
                            )
                            (in desk)
                            (agent_holds ?b ?v)
                          )
                          (in ?v)
                        )
                      )
                      (game-conserved
                        (= (distance ?v ?b) 1)
                      )
                      (exists (?p - game_object)
                        (and
                          (exists (?o - golfball)
                            (game-optional
                              (agent_holds ?o)
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (and
                  (game-conserved
                    (agent_holds ?b ?b)
                  )
                )
              )
              (forall (?c - hexagonal_bin)
                (and
                  (forall (?f - hexagonal_bin)
                    (and
                      (exists (?o - (either beachball desktop))
                        (and
                          (forall (?n - hexagonal_bin)
                            (and
                              (game-optional
                                (and
                                  (and
                                    (and
                                      (agent_holds ?c)
                                      (and
                                        (on floor)
                                        (on ?o ?n)
                                      )
                                    )
                                    (agent_holds ?n ?o)
                                  )
                                  (agent_holds ?b)
                                )
                              )
                              (exists (?l - hexagonal_bin)
                                (game-conserved
                                  (and
                                    (and
                                      (in_motion ?o ?n)
                                      (in ?n top_shelf)
                                    )
                                    (adjacent ?n)
                                  )
                                )
                              )
                              (game-optional
                                (in_motion ?c ?f)
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                  (game-optional
                    (on bridge_block)
                  )
                )
              )
              (and
                (and
                  (and
                    (forall (?a - doggie_bed)
                      (exists (?d - hexagonal_bin ?c - chair)
                        (game-conserved
                          (in_motion ?a)
                        )
                      )
                    )
                  )
                )
                (forall (?u - (either chair))
                  (and
                    (game-optional
                      (in_motion ?u ?u ?b)
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?i - wall ?j - hexagonal_bin ?j - hexagonal_bin ?s - dodgeball)
        (exists (?w - dodgeball)
          (exists (?q - hexagonal_bin)
            (then
              (once (in_motion agent ?q) )
              (hold (in_motion pink) )
              (once (exists (?n - (either credit_card key_chain)) (and (not (not (on ?n) ) ) (and (same_color ?s agent) (same_color ?q) ) ) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?g - hexagonal_bin)
        (exists (?j - hexagonal_bin ?q ?a - game_object)
          (then
            (once (and (exists (?d - hexagonal_bin) (agent_holds floor sideways) ) (not (and (< (distance ?q 8) 1) (and (in_motion ?q ?g ?g) (on ?g) (type ?a ?q) (agent_holds agent ?q) ) (on ?g) (is_setup_object agent) ) ) ) )
            (once (not (or (and (and (agent_holds ?a rug) (agent_holds ?a ?q) ) (or (agent_holds ?a) (< (distance 6) (distance ?q ?g)) ) (not (in ?g ?a) ) ) (or (< 0.5 0.5) (in ?q) ) ) ) )
            (once (on ?q) )
          )
        )
      )
    )
  )
)
(:terminal
  (= 30 (count-nonoverlapping preferenceA:basketball) )
)
(:scoring
  3
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?s - dodgeball)
        (then
          (once (not (and (not (in_motion ?s) ) (not (agent_holds ?s) ) ) ) )
          (hold (in_motion ?s) )
          (once (and (is_setup_object ?s ?s) (exists (?d - block) (in_motion ?s ?d) ) (and (agent_holds ?s) (agent_holds ?s) ) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (hold (not (agent_holds ?xxx ?xxx) ) )
        (once (on east_sliding_door ?xxx) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
    (preference preferenceC
      (then
        (once (in_motion ?xxx) )
        (hold-for 1 (on desk) )
        (once (on ?xxx) )
      )
    )
    (preference preferenceD
      (exists (?x ?v - building ?z - doggie_bed)
        (exists (?n - hexagonal_bin ?j - (either hexagonal_bin cellphone))
          (exists (?w - building)
            (then
              (hold-while (in_motion ?z ?j) (on rug ?j) )
              (once (not (in_motion ?j) ) )
              (hold (agent_holds ?z ?z) )
            )
          )
        )
      )
    )
    (preference preferenceE
      (then
        (once (agent_holds ?xxx) )
        (hold-while (and (exists (?s - chair) (on ?s) ) (faces ?xxx ?xxx) ) (in bed ?xxx) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
    (preference preferenceF
      (exists (?j - (either cd alarm_clock) ?r - hexagonal_bin ?m - curved_wooden_ramp ?u - hexagonal_bin)
        (exists (?h - hexagonal_bin ?e - pyramid_block)
          (then
            (once (in_motion ?u) )
            (once (and (is_setup_object ?u front) (and (in_motion ?e ?e) (not (and (agent_holds ?u ?e) (in desk) (< 1 (x_position ?u 3)) (or (in_motion ?e) (and (in_motion ?u ?u ?u) (and (exists (?i - dodgeball ?p ?k ?j - dodgeball ?r ?f - curved_wooden_ramp ?b - building) (<= 2 5) ) (< 2 (distance ?e agent)) ) ) ) (on ?e) (not (not (in ?u) ) ) (in_motion ?e) (exists (?o - (either hexagonal_bin bridge_block)) (on ?o) ) ) ) ) ) )
            (hold (in_motion ?u) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-same-positions preferenceC:yellow) (count-shortest preferenceE:triangle_block:dodgeball) )
    (>= (+ (count-nonoverlapping preferenceC:side_table) (/ (* (count-once-per-objects preferenceF:cube_block:pink) (count-nonoverlapping preferenceD:orange) )
          10
        )
        (* (* (* 1 (+ 3 (count-nonoverlapping-measure preferenceD:dodgeball:pink) (count-nonoverlapping preferenceE:golfball:dodgeball) (* (* (count-nonoverlapping preferenceA:beachball) (count-once-per-objects preferenceF:cube_block) )
                  40
                )
              )
            )
            15
          )
          (external-forall-maximize
            (not
              (count-nonoverlapping preferenceB:pink)
            )
          )
        )
      )
      6
    )
  )
)
(:scoring
  (count-unique-positions preferenceC:golfball:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?c - teddy_bear ?h - chair)
    (or
      (game-optional
        (or
          (in_motion ?h ?h)
          (in ?h ?h)
          (agent_holds ?h ?h)
        )
      )
      (exists (?j - hexagonal_bin)
        (and
          (exists (?q - hexagonal_bin ?m - hexagonal_bin)
            (game-optional
              (type ?m ?m)
            )
          )
        )
      )
      (and
        (game-conserved
          (in ?h)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b - block)
        (then
          (once (and (not (in rug) ) (on ?b) (in_motion ?b) ) )
          (hold (in ?b) )
          (hold (and (and (exists (?s - cube_block ?o - cube_block) (touch ?o ?b) ) (agent_holds ?b) ) (agent_holds ?b) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (in_motion ?xxx) )
        (hold-while (and (exists (?v - teddy_bear ?v ?i - bridge_block) (and (agent_holds ?v ?v) (in agent) (equal_x_position ?i) (and (on ?i) (object_orientation bridge_block ?i) ) (not (in_motion ?v) ) (and (and (on ?v) (agent_holds ?i) ) (same_color ?v) (agent_holds ?i) ) (not (same_color ?i ?v) ) (agent_holds agent green) (on ?i) (and (on agent ?v) (in_motion agent ?v) ) ) ) (on ?xxx ?xxx desk ?xxx) ) (and (agent_holds desk) (= (distance 7 ?xxx) (distance bed ?xxx)) ) (in_motion ?xxx) )
        (hold (not (not (not (in_motion ?xxx ?xxx) ) ) ) )
      )
    )
  )
)
(:terminal
  (>= 10 (count-nonoverlapping preferenceB:golfball) )
)
(:scoring
  (* (* (- (count-nonoverlapping preferenceA:pink_dodgeball) )
      (count-nonoverlapping-measure preferenceB:basketball)
      (total-time)
      (or
        (count-once-per-objects preferenceA:cube_block)
      )
      7
      (+ 1 (count-nonoverlapping preferenceA:yellow) )
    )
    4
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (at-end
        (and
          (touch ?xxx ?xxx)
          (and
            (and
              (on ?xxx ?xxx)
              (not
                (exists (?z - hexagonal_bin)
                  (exists (?c - curved_wooden_ramp)
                    (in_motion rug ?c)
                  )
                )
              )
            )
            (and
              (not
                (same_color rug)
              )
            )
          )
          (agent_holds ?xxx)
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (= 6 (* (external-forall-maximize (count-nonoverlapping preferenceA:hexagonal_bin) ) (/ (count-nonoverlapping preferenceA:blue_dodgeball:yellow_cube_block) (* (count-once-per-objects preferenceA:red) (count-once preferenceA:golfball) )
          )
        )
      )
    )
    (>= (+ 30 (count-nonoverlapping preferenceA:purple) )
      3
    )
  )
)
(:scoring
  (count-nonoverlapping preferenceA:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?g - dodgeball)
    (and
      (game-conserved
        (agent_holds ?g)
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - teddy_bear)
      (and
        (preference preferenceA
          (at-end
            (and
              (not
                (not
                  (on ?k)
                )
              )
              (agent_holds ?k ?k)
              (in_motion ?k)
            )
          )
        )
        (preference preferenceB
          (exists (?y - blinds ?s ?e - yellow_cube_block)
            (then
              (once (and (and (agent_holds ?s ?s) (forall (?b ?i - hexagonal_bin) (in_motion ?b ?k) ) (touch ?k top_shelf) ) (between ?s) ) )
              (once (in_motion ?s) )
              (any)
            )
          )
        )
        (preference preferenceC
          (exists (?b - golfball ?q - cube_block)
            (exists (?x ?b - doggie_bed)
              (then
                (once (in desk ?x) )
                (hold (on ?k) )
                (once (in_motion ?x ?x) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (count-once-per-objects preferenceA:golfball) )
)
(:scoring
  (* (+ 1 7 )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?e - (either chair cd cellphone))
    (game-optional
      (and
        (agent_holds ?e)
        (in ?e ?e)
        (in_motion ?e)
        (type ?e)
        (in_motion ?e)
        (exists (?i - dodgeball)
          (adjacent ?e ?i ?i)
        )
        (not
          (and
            (agent_holds ?e)
            (not
              (not
                (forall (?t - teddy_bear)
                  (not
                    (and
                      (not
                        (and
                          (in_motion right)
                          (and
                            (not
                              (agent_holds ?t)
                            )
                            (in ?t)
                          )
                          (not
                            (and
                              (not
                                (and
                                  (in_motion ?t)
                                  (not
                                    (in ?e ?e)
                                  )
                                )
                              )
                              (touch ?t agent)
                              (agent_holds pink_dodgeball ?e)
                            )
                          )
                        )
                      )
                      (in_motion ?e)
                    )
                  )
                )
              )
            )
          )
        )
        (in ?e)
        (in_motion bed ?e)
        (or
          (agent_holds ?e sideways)
        )
        (not
          (agent_holds rug)
        )
        (agent_holds ?e)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?a - hexagonal_bin)
        (exists (?k - ball ?t - curved_wooden_ramp)
          (exists (?q - dodgeball ?b - building)
            (exists (?i - teddy_bear)
              (exists (?j - dodgeball)
                (at-end
                  (and
                    (and
                      (adjacent bed)
                      (exists (?o - ball)
                        (in_motion ?a)
                      )
                    )
                    (and
                      (< (distance room_center ?j) 1)
                      (and
                        (and
                          (agent_holds agent ?a)
                          (not
                            (not
                              (< (distance 9 3) 1)
                            )
                          )
                        )
                        (not
                          (in_motion agent)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?l - cylindrical_block)
        (then
          (once (< 1 (distance 7 ?l)) )
          (once (agent_holds south_wall ?l) )
          (once (not (game_start ?l) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count-nonoverlapping preferenceA:hexagonal_bin) (* (+ (count-nonoverlapping preferenceA) 8 )
        7
      )
    )
    (count-nonoverlapping preferenceA:purple:book)
  )
)
(:scoring
  (+ (count-once-per-objects preferenceB:basketball) (+ (count-shortest preferenceA:pink) 5 (* 3 )
      (>= (count-nonoverlapping preferenceA:dodgeball:dodgeball) (count-once-per-objects preferenceA:beachball:pink_dodgeball) )
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (not
      (and
        (and
          (in_motion ?xxx)
          (not
            (not
              (in_motion ?xxx)
            )
          )
        )
        (and
          (agent_holds back ?xxx)
          (and
            (agent_holds ?xxx)
            (adjacent ?xxx)
            (not
              (not
                (< 10 (distance ?xxx 5))
              )
            )
          )
          (agent_holds agent ?xxx)
          (agent_holds ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (in_motion ?xxx) )
        (hold (and (same_type ?xxx) (< 1 1) ) )
        (hold-while (and (in_motion ?xxx) (agent_holds bed agent) ) (agent_holds ?xxx) (in_motion ?xxx) )
      )
    )
    (preference preferenceB
      (then
        (once (agent_holds ?xxx) )
        (once (exists (?p - cube_block ?t - cube_block ?r - ball) (in_motion ?r ?r) ) )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preferenceA:orange) 5 )
)
(:scoring
  10
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?w - (either lamp dodgeball key_chain) ?y - (either book red))
    (game-conserved
      (in_motion ?y desk ?y)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b ?h - ball ?m - (either dodgeball cube_block) ?r - red_pyramid_block)
        (at-end
          (agent_holds ?r)
        )
      )
    )
  )
)
(:terminal
  (and
    (> (* 2 (count-unique-positions preferenceA:dodgeball) )
      (count-nonoverlapping preferenceA:basketball)
    )
    (> (* 3 (total-score) )
      (external-forall-maximize
        3
      )
    )
  )
)
(:scoring
  (count-nonoverlapping preferenceA:basketball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?a ?c ?z - pyramid_block ?d - hexagonal_bin)
    (game-conserved
      (exists (?f - block ?v - doggie_bed ?i - hexagonal_bin)
        (in_motion front ?i)
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - cube_block)
      (and
        (preference preferenceA
          (then
            (once (and (not (same_object ?b ?b) ) (on ?b ?b) ) )
            (once (agent_holds ?b) )
            (hold (not (adjacent agent ?b) ) )
          )
        )
        (preference preferenceB
          (then
            (hold (on ?b) )
            (hold (in_motion ?b) )
            (once (not (not (in_motion ?b) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preferenceA:basketball:golfball) 30 )
)
(:scoring
  (count-once-per-objects preferenceB:beachball:purple)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (exists (?a - game_object)
      (and
        (game-optional
          (in_motion ?a ?a)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?g - doggie_bed)
        (at-end
          (on ?g)
        )
      )
    )
    (preference preferenceB
      (exists (?l - dodgeball ?p - hexagonal_bin)
        (then
          (hold-while (not (and (not (and (not (agent_holds ?p ?p) ) (in ?p) ) ) (touch ?p) ) ) (agent_holds ?p ?p) )
          (once (not (agent_holds ?p) ) )
          (hold (and (in_motion agent) (not (and (and (agent_holds ?p) (in_motion ?p) ) (above ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (external-forall-maximize (* (total-time) (total-time) )
      )
      4
    )
    (>= 10 5 )
  )
)
(:scoring
  (count-nonoverlapping preferenceA:red:blue_pyramid_block)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (and
        (game-conserved
          (and
            (in_motion ?xxx ?xxx)
            (agent_holds floor ?xxx)
          )
        )
        (game-conserved
          (and
            (< 1 (distance ?xxx ?xxx))
            (in_motion agent tan)
            (exists (?a - cube_block ?p - dodgeball)
              (exists (?z - dodgeball)
                (and
                  (and
                    (not
                      (and
                        (in_motion ?z)
                        (not
                          (in_motion ?p)
                        )
                        (agent_holds ?z ?p)
                        (game_start ?p rug)
                      )
                    )
                    (>= (distance room_center ?z) (distance ?z ?z))
                    (not
                      (in_motion ?z rug)
                    )
                  )
                  (in ?p)
                  (< 1 (distance room_center agent))
                  (agent_holds top_drawer ?p)
                )
              )
            )
            (in ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?i - game_object)
        (then
          (once (and (not (in_motion ?i) ) (forall (?n - cube_block) (in_motion ?n ?n) ) ) )
          (once (same_color ?i ?i) )
          (once (not (in bed ?i) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (not (not (in ?xxx) ) ) )
        (once (agent_holds ?xxx ?xxx) )
        (once (in ?xxx ?xxx) )
      )
    )
    (preference preferenceC
      (then
        (once (exists (?c - curved_wooden_ramp ?c - beachball) (not (and (on ?c) (rug_color_under desk) (in floor) ) ) ) )
        (hold (not (and (and (object_orientation ?xxx ?xxx) (or (agent_holds rug ?xxx) (on ?xxx) ) (and (on ?xxx) (not (in_motion ) ) ) ) (agent_holds west_wall) (in_motion ?xxx ?xxx) (>= 0 1) ) ) )
        (once (and (on ?xxx) (and (on ?xxx) (on ?xxx ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (>= (count-nonoverlapping preferenceB:blue_pyramid_block:blue_cube_block) (count-once-per-objects preferenceA:pink_dodgeball:book) )
)
(:scoring
  (count-overlapping preferenceC:dodgeball:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-optional
      (not
        (in ?xxx ?xxx)
      )
    )
    (and
      (game-conserved
        (and
          (not
            (< 10 (distance ?xxx ?xxx))
          )
          (and
            (and
              (in_motion ?xxx)
              (forall (?u - hexagonal_bin ?k - (either cylindrical_block pencil))
                (agent_holds ?k ?k)
              )
            )
            (on ?xxx brown)
            (not
              (rug_color_under agent)
            )
            (in_motion ?xxx ?xxx)
            (object_orientation ?xxx)
            (rug_color_under ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?g ?l - game_object)
        (then
          (once (and (and (not (is_setup_object ?g ?g) ) (same_color ?g) (touch brown) ) (not (agent_holds ?g) ) ) )
          (hold (agent_holds agent) )
          (once (not (not (in_motion ?g ?l) ) ) )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (* 100 3 )
      (* (count-nonoverlapping preferenceA:blue_cube_block) (* (count-once-per-objects preferenceA:hexagonal_bin) 1 )
        (count-nonoverlapping preferenceA:pink)
      )
    )
    (and
      (>= (* (count-nonoverlapping preferenceA:golfball) (count-nonoverlapping preferenceA:red) )
        (+ 2 (external-forall-maximize (total-time) ) )
      )
    )
    (= (* (count-same-positions preferenceA:blue_dodgeball) 15 )
      5
    )
  )
)
(:scoring
  (count-overlapping preferenceA:dodgeball:yellow)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (and
      (in_motion agent ?xxx)
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold (agent_holds ?xxx ?xxx) )
        (once (in_motion ?xxx) )
        (hold (not (on ?xxx) ) )
      )
    )
  )
)
(:terminal
  (not
    (>= (* 3 16 )
      (count-once-per-objects preferenceA:hexagonal_bin:dodgeball)
    )
  )
)
(:scoring
  2
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold (same_object ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
        (once (and (< (distance 5 room_center) 1) (not (touch ?xxx) ) (not (open upside_down ?xxx ?xxx) ) (and (in_motion ?xxx agent) (not (on ?xxx ?xxx) ) ) (in ?xxx bed) (exists (?k - teddy_bear ?j - building ?q - ball) (in_motion ?q ?q) ) (in_motion bed agent) ) )
      )
    )
    (preference preferenceB
      (then
        (hold (on ?xxx) )
        (hold (and (in_motion ?xxx) (agent_holds ?xxx) (and (and (not (agent_holds ?xxx) ) (not (in ?xxx ?xxx) ) ) (and (not (agent_holds ?xxx ?xxx) ) (exists (?w - ball ?r - red_pyramid_block ?c ?j - (either mug yellow_cube_block)) (same_color ?c) ) (on ?xxx agent) ) ) (and (touch floor) (agent_holds agent) ) (agent_holds ?xxx) (agent_holds agent) ) )
        (hold (not (agent_holds agent ?xxx) ) )
      )
    )
  )
)
(:terminal
  (> (count-nonoverlapping preferenceB:yellow:yellow) (- 10 )
  )
)
(:scoring
  (* 5 (or 10 (count-once-per-objects preferenceB:dodgeball) ) )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?a - drawer ?t - hexagonal_bin)
    (and
      (game-conserved
        (agent_holds ?t ?t)
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - pillow)
      (and
        (preference preferenceA
          (exists (?v - (either cube_block golfball))
            (exists (?m - (either golfball yellow_cube_block))
              (exists (?e - cube_block)
                (exists (?o - (either hexagonal_bin golfball key_chain))
                  (exists (?j - hexagonal_bin ?c - green_triangular_ramp ?f - chair ?g - blue_cube_block)
                    (then
                      (once (in_motion pink_dodgeball) )
                      (hold (agent_holds desk) )
                      (once (in_motion upright) )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count-nonoverlapping preferenceA:beachball) (* 20 (count-once-per-objects preferenceA:pink) )
  )
)
(:scoring
  (count-nonoverlapping preferenceA:red)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?q - golfball)
    (forall (?i - game_object)
      (game-conserved
        (agent_holds ?i ?i)
      )
    )
  )
)
(:constraints
  (and
    (forall (?p - dodgeball)
      (and
        (preference preferenceA
          (exists (?j - tall_cylindrical_block ?c ?n - dodgeball)
            (then
              (once (and (in_motion ?p ?p) (agent_holds front) ) )
              (hold-to-end (not (in_motion ?n ?c) ) )
              (once (not (and (exists (?h ?f - block) (agent_holds ?f) ) (and (not (in_motion rug) ) (and (in_motion agent ?n) (adjacent ?p floor) ) ) ) ) )
            )
          )
        )
        (preference preferenceB
          (exists (?b - hexagonal_bin)
            (exists (?v - dodgeball ?n ?w ?l ?r ?h ?v - hexagonal_bin)
              (at-end
                (and
                  (in_motion sideways ?r)
                  (in_motion ?w ?l)
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?l - ball ?k - cube_block)
        (exists (?h - (either triangular_ramp))
          (then
            (once (same_color agent) )
            (once (agent_holds ?h) )
            (once (in_motion ?k) )
          )
        )
      )
    )
    (preference preferenceD
      (at-end
        (in_motion ?xxx)
      )
    )
  )
)
(:terminal
  (>= (not 3 ) (- (count-once-per-objects preferenceA:doggie_bed:tall_cylindrical_block:dodgeball) (count-once-per-objects preferenceA) ) )
)
(:scoring
  (* (+ 12 (- 18 )
    )
    (count-once-per-objects preferenceD:pyramid_block)
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-conserved
      (agent_holds bottom_shelf)
    )
    (and
      (exists (?e - chair ?g - (either pyramid_block book laptop))
        (forall (?x - cube_block ?j - teddy_bear)
          (and
            (game-conserved
              (and
                (agent_holds ?j)
                (agent_holds ?j ?g)
              )
            )
            (game-optional
              (not
                (and
                  (not
                    (not
                      (adjacent agent)
                    )
                  )
                  (not
                    (in_motion desk)
                  )
                )
              )
            )
          )
        )
      )
    )
    (forall (?e - dodgeball)
      (forall (?x - hexagonal_bin)
        (and
          (and
            (game-optional
              (agent_holds ?e ?e)
            )
            (forall (?b - cube_block)
              (and
                (and
                  (and
                    (game-optional
                      (< 1 10)
                    )
                  )
                  (and
                    (exists (?h - doggie_bed ?p - block ?m - hexagonal_bin)
                      (game-conserved
                        (< 2 (distance 5 agent))
                      )
                    )
                    (exists (?v - shelf)
                      (exists (?p - hexagonal_bin)
                        (and
                          (and
                            (game-conserved
                              (agent_holds ?e)
                            )
                          )
                        )
                      )
                    )
                  )
                  (game-conserved
                    (in_motion ?x)
                  )
                )
                (and
                  (game-optional
                    (not
                      (touch ?b)
                    )
                  )
                  (forall (?t - hexagonal_bin)
                    (exists (?g - cube_block ?k ?q ?z - cube_block)
                      (and
                        (exists (?i - game_object)
                          (forall (?v - game_object)
                            (and
                              (and
                                (and
                                  (game-optional
                                    (in_motion ?k)
                                  )
                                  (game-conserved
                                    (agent_holds ?b)
                                  )
                                )
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (game-optional
                  (in ?e)
                )
              )
            )
            (game-conserved
              (not
                (touch ?e)
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?p - game_object)
        (exists (?q - hexagonal_bin ?q - hexagonal_bin ?f - cube_block ?t - building)
          (then
            (any)
            (hold (agent_holds ?p) )
            (once (= (distance ?p ?p) (distance ?t ?t)) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-nonoverlapping-measure preferenceA:brown) (* 100 5 )
  )
)
(:scoring
  10
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (not
    (game-conserved
      (not
        (and
          (in_motion ?xxx ?xxx)
          (agent_holds agent)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold (and (< 3 (distance_side )) (agent_holds agent ?xxx) ) )
        (hold (and (in ?xxx ?xxx) (and (same_color back agent) (and (agent_holds ?xxx) (exists (?j - block) (agent_holds pink_dodgeball) ) (in_motion ?xxx ?xxx) ) ) ) )
        (hold (equal_x_position desk ?xxx) )
      )
    )
    (preference preferenceB
      (forall (?m - (either basketball key_chain) ?y - chair ?o - (either cylindrical_block golfball teddy_bear))
        (at-end
          (in_motion ?o)
        )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preferenceB) 4 )
)
(:scoring
  (* (- (* (* 20 1 )
        (count-nonoverlapping preferenceB:doggie_bed)
      )
    )
    (+ (+ (count-nonoverlapping preferenceA:dodgeball) (count-once preferenceB:dodgeball) (count-shortest preferenceB:dodgeball) )
      2
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?y - dodgeball ?r - cube_block)
    (game-conserved
      (not
        (agent_holds upright)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b - doggie_bed)
        (exists (?k - cube_block)
          (exists (?a - pillow)
            (then
              (once (in ?a) )
              (hold (on ?b) )
              (once (not (in_motion ?a ?k) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-nonoverlapping preferenceA:blue_cube_block) (* (count-once preferenceA:top_drawer) (and (* (* (count-once preferenceA:green) (count-nonoverlapping preferenceA:alarm_clock) )
        )
        2
        (count-nonoverlapping preferenceA:side_table)
      )
      (count-nonoverlapping preferenceA:dodgeball:dodgeball)
    )
  )
)
(:scoring
  10
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (forall (?y - (either lamp triangle_block tall_cylindrical_block) ?d - wall ?r - (either dodgeball laptop) ?l - block ?e - cube_block)
      (game-conserved
        (in ?e ?e)
      )
    )
    (exists (?e - (either dodgeball book))
      (and
        (game-optional
          (on ?e)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?k - golfball)
        (exists (?i - ball)
          (at-end
            (on ?i)
          )
        )
      )
    )
    (preference preferenceB
      (exists (?c - golfball)
        (exists (?q ?n ?a - triangular_ramp)
          (exists (?d ?p - cube_block ?l - hexagonal_bin ?g - ball ?p - cube_block)
            (at-end
              (agent_holds ?n yellow)
            )
          )
        )
      )
    )
    (preference preferenceC
      (then
        (once (and (faces ?xxx ?xxx ?xxx) (agent_holds ?xxx) ) )
        (hold (= 0 (distance_side ?xxx ?xxx)) )
      )
    )
  )
)
(:terminal
  (>= 15 (+ (* (count-nonoverlapping preferenceC:dodgeball) (- 2 )
      )
      2
    )
  )
)
(:scoring
  (external-forall-maximize
    3
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (< 7 (distance 9 agent))
  )
)
(:constraints
  (and
    (forall (?l - desk_shelf)
      (and
        (preference preferenceA
          (then
            (once (in bed) )
            (hold-to-end (and (adjacent rug) (not (not (and (exists (?r ?s - cube_block ?s - hexagonal_bin) (in ?s) ) (on ?l ?l) (and (agent_holds ?l rug) (agent_holds ?l) ) ) ) ) (< (distance 10 ?l) (distance agent)) ) )
            (once (in_motion ?l ?l) )
          )
        )
        (preference preferenceB
          (exists (?j - cube_block ?x - dodgeball)
            (exists (?j - hexagonal_bin ?i - building ?b - hexagonal_bin ?g - hexagonal_bin)
              (then
                (once (and (= 1 (distance ?l 10)) (in_motion ?x ?l) (and (and (not (in_motion ?l) ) (not (agent_holds ?x) ) ) (on ?x) ) ) )
                (once (or (not (adjacent ?l) ) (adjacent brown) ) )
                (once (and (adjacent_side ?l ?l) (<= 6 1) ) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (not (count-total preferenceA:beachball:dodgeball) ) (count-nonoverlapping preferenceA:yellow_cube_block) )
)
(:scoring
  (total-score)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (exists (?j ?p ?w - tan_cube_block ?n - curved_wooden_ramp)
      (exists (?m - dodgeball)
        (forall (?b - curved_wooden_ramp ?f - (either mug flat_block dodgeball))
          (and
            (and
              (game-optional
                (and
                  (and
                    (in_motion top_shelf ?n)
                    (in_motion ?m ?f)
                  )
                  (in_motion ?m ?n)
                )
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?c - ball ?a ?n - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?w - building ?i - building ?l - dodgeball)
            (exists (?u - triangular_ramp ?x ?g ?k - chair ?v - (either main_light_switch alarm_clock credit_card blue_cube_block))
              (exists (?d - (either bridge_block book))
                (exists (?m - hexagonal_bin ?k - doggie_bed)
                  (exists (?p - game_object ?e - hexagonal_bin)
                    (exists (?i - hexagonal_bin)
                      (exists (?m ?w ?s - shelf)
                        (then
                          (hold-to-end (agent_holds ?a) )
                          (hold-while (between ?w) (toggled_on agent) )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 3 )
)
(:scoring
  (count-once-per-objects preferenceA:red)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (and
        (forall (?a - block)
          (and
            (forall (?q - game_object)
              (exists (?i - block ?j - building)
                (game-conserved
                  (not
                    (agent_holds ?j)
                  )
                )
              )
            )
            (forall (?r - block)
              (exists (?j - chair ?s - ball ?f - blinds)
                (game-conserved
                  (in ?f)
                )
              )
            )
          )
        )
        (game-optional
          (and
            (agent_holds ?xxx ?xxx)
            (and
              (in_motion color ?xxx)
              (on ?xxx ?xxx)
              (above ?xxx)
              (not
                (and
                  (exists (?i - cube_block)
                    (toggled_on ?i)
                  )
                  (game_start ?xxx)
                )
              )
            )
          )
        )
      )
    )
    (and
      (and
        (game-conserved
          (not
            (agent_holds agent)
          )
        )
        (game-conserved
          (and
            (agent_holds ?xxx)
            (in_motion ?xxx)
          )
        )
      )
    )
    (game-optional
      (and
        (in ?xxx)
        (not
          (in_motion south_wall ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - dodgeball)
      (and
        (preference preferenceA
          (exists (?j - block)
            (exists (?m - dodgeball)
              (exists (?q - hexagonal_bin ?y - dodgeball)
                (exists (?z - hexagonal_bin ?g - bridge_block)
                  (exists (?z - (either dodgeball golfball pyramid_block))
                    (exists (?q - (either pyramid_block basketball dodgeball))
                      (exists (?e - game_object)
                        (exists (?h - dodgeball)
                          (then
                            (hold (on ?e) )
                            (once (in_motion ?j) )
                            (once (in_motion bed ?g ?y) )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (preference preferenceB
          (then
            (once (and (not (in_motion ?d ?d bed) ) (in_motion bed) (in_motion ?d) (exists (?a - (either triangle_block pyramid_block)) (not (in ?a ?d) ) ) (in rug desk) (and (on ?d ?d) (in_motion ?d) ) ) )
            (once (not (agent_holds ?d) ) )
            (once (adjacent_side ?d) )
          )
        )
      )
    )
  )
)
(:terminal
  (> 4 (* 6 (* 3 6 )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (< 7 7)
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (not (in_motion agent blue) ) )
        (once (exists (?o - hexagonal_bin) (in ?o) ) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count-nonoverlapping preferenceA:yellow:purple:side_table) 2 )
)
(:scoring
  (count-nonoverlapping preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (forall (?n - building ?o - block)
      (and
        (not
          (and
            (game-conserved
              (in_motion ?o ?o)
            )
          )
        )
        (exists (?e - game_object ?b ?j - wall)
          (game-conserved
            (not
              (adjacent ?j)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (same_object ?xxx ?xxx) )
        (once (and (in_motion ?xxx) (in_motion ?xxx) (and (exists (?h - (either yellow_cube_block pencil) ?y - blinds) (and (agent_holds ?y) (not (and (and (and (< 1 1) ) (not (agent_holds ?y ?y) ) ) (adjacent ?y) ) ) ) ) (in_motion ?xxx) ) (adjacent_side ?xxx) (= (distance ) (distance ?xxx 8)) (< (distance door ?xxx ?xxx) (distance ?xxx ?xxx)) (same_color ?xxx) (and (= 1 7) (on bed ?xxx) ) ) )
        (once (same_object agent) )
      )
    )
    (preference preferenceB
      (then
        (hold (agent_holds ?xxx ?xxx) )
        (hold (adjacent ?xxx) )
        (once (on rug ?xxx) )
      )
    )
    (preference preferenceC
      (at-end
        (in_motion ?xxx)
      )
    )
  )
)
(:terminal
  (>= 5 10 )
)
(:scoring
  6
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?n - hexagonal_bin)
    (exists (?r ?j - bridge_block)
      (and
        (game-optional
          (in_motion ?j)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?m - hexagonal_bin ?y - chair)
      (and
        (preference preferenceA
          (then
            (hold-for 8 (and (< (distance ?y 2) 4) (agent_holds ?y) ) )
            (once (in_motion ?y ?y) )
            (once (agent_holds ?y) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (+ (+ (count-once-per-objects preferenceA:blue_pyramid_block) (count-nonoverlapping preferenceA:hexagonal_bin) )
        2
      )
      (* (+ (+ (* (+ 3 (and 3 2 ) )
              (-
                (total-score)
                (- 20 )
              )
              (count-once preferenceA:pink_dodgeball)
              8
              5
              2
            )
            (count-nonoverlapping preferenceA)
          )
          2
        )
        (count-unique-positions preferenceA:basketball:alarm_clock)
      )
    )
  )
)
(:scoring
  10
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (exists (?v - cube_block)
        (game-optional
          (open ?v ?v)
        )
      )
      (and
        (game-conserved
          (and
            (in_motion ?xxx ?xxx)
            (forall (?z - red_dodgeball)
              (agent_holds east_sliding_door)
            )
          )
        )
      )
      (forall (?m - (either golfball cellphone))
        (game-conserved
          (agent_holds ?m ?m)
        )
      )
    )
    (game-conserved
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?m - block ?i - cube_block)
        (then
          (hold (in_motion ?i) )
          (hold (not (agent_holds ?i) ) )
          (hold-to-end (and (agent_holds back green_golfball ?i) (on color rug) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (and (not (agent_holds ?xxx ?xxx) ) (= (distance ?xxx ?xxx ?xxx) (distance ?xxx)) ) )
        (hold (not (not (agent_holds desk) ) ) )
        (once (and (on ?xxx) ) )
      )
    )
  )
)
(:terminal
  (or
    (>= 15 (external-forall-maximize 20 ) )
    (>= (count-nonoverlapping preferenceB:dodgeball) (- (count-once-per-objects preferenceA:golfball) )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?o - hexagonal_bin)
    (or
      (exists (?z - cube_block ?j - hexagonal_bin)
        (game-optional
          (agent_holds agent ?o)
        )
      )
      (game-conserved
        (touch ?o agent)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (agent_holds ?xxx) )
        (once (in_motion ?xxx) )
        (hold (and (not (on ?xxx ?xxx) ) (on ?xxx desk) ) )
      )
    )
    (forall (?j - hexagonal_bin ?l ?e ?r ?j ?x ?g - hexagonal_bin)
      (and
        (preference preferenceB
          (then
            (hold (in_motion ?l) )
            (once (adjacent pink_dodgeball) )
            (once (touch ?e ?e) )
          )
        )
        (preference preferenceC
          (then
            (hold (on ?j ?j) )
            (once (agent_holds ?r) )
            (hold-while (not (in_motion ?x) ) (on ?g) )
          )
        )
        (preference preferenceD
          (exists (?o - ball)
            (then
              (once (< 0.5 0) )
              (once (and (touch ?j ?g) (in_motion ?o) ) )
              (hold-while (and (not (and (not (touch ?x) ) (touch ?l) ) ) (not (touch ?o) ) ) (and (< (distance ?x 9) 1) (agent_holds bed) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- (* (count-nonoverlapping preferenceB:dodgeball) (+ (count-nonoverlapping-measure preferenceA:dodgeball:green) 3 )
      )
    )
    (count-nonoverlapping preferenceB:basketball:hexagonal_bin:red)
  )
)
(:scoring
  (external-forall-minimize
    10
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?m - hexagonal_bin)
    (game-conserved
      (and
        (and
          (and
            (< 2 (distance desk ?m))
          )
          (and
            (agent_holds ?m bed)
          )
        )
        (< (distance ?m 0) (distance agent ?m ?m))
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b - doggie_bed)
        (exists (?v - curved_wooden_ramp)
          (exists (?j - hexagonal_bin ?d - hexagonal_bin ?j - game_object)
            (then
              (hold (and (touch ?j) (in_motion ?v) ) )
              (hold (not (agent_holds desk ?b) ) )
              (hold-while (on upright ?j) (not (not (and (adjacent ?j bridge_block) (not (not (or (and (exists (?u - (either cylindrical_block teddy_bear)) (and (not (agent_holds ?j) ) (touch ?u) (in ?b agent) ) ) (and (or (in ?j ?j) (in_motion ?j) ) (in_motion pink) ) (on desk) (on ?j) ) (in ?v) (in_motion ?b desk) (not (and (in_motion rug floor) (in_motion ?j) (not (agent_holds ?v ?b ?b ?b) ) ) ) ) ) ) (and (in_motion ?b brown) (agent_holds ?v) ) ) ) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?b - game_object)
        (then
          (hold (and (on ?b ?b) (adjacent ?b) ) )
          (once (adjacent ?b rug) )
          (once (not (in_motion ?b ?b) ) )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (* (count-nonoverlapping preferenceB:hexagonal_bin) 3 )
      (* (total-score) 3 (* 2 7 )
        (total-time)
      )
    )
  )
)
(:scoring
  (count-once-per-external-objects preferenceB:beachball:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (forall-sequence (?d - red_pyramid_block)
          (then
            (once (in ?d agent) )
            (once (not (and (in ) (agent_holds ?d) ) ) )
            (hold-while (agent_holds ?d) (not (in_motion ?d) ) )
          )
        )
        (once (object_orientation ?xxx) )
        (once (in_motion ?xxx) )
      )
    )
    (preference preferenceB
      (exists (?o - (either dodgeball floor))
        (exists (?r - (either dodgeball tall_cylindrical_block) ?b - (either cellphone top_drawer) ?m - (either book floor) ?j ?a - doggie_bed ?d - teddy_bear)
          (exists (?e - (either yellow_cube_block cellphone))
            (exists (?h - hexagonal_bin)
              (then
                (hold-while (not (< 1 1) ) (not (not (touch ?e ?h) ) ) )
                (hold (agent_holds ?d) )
                (hold (not (game_start ?e rug) ) )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (then
        (hold (same_color ?xxx ?xxx) )
        (once (and (and (in floor) (on ?xxx ?xxx) ) (game_over ?xxx ?xxx bed) (in agent ?xxx) ) )
        (once (and (and (and (in_motion ?xxx) (and (adjacent floor ?xxx) (not (and (not (and (not (< (distance side_table ?xxx) 6) ) (not (not (and (not (on ?xxx ?xxx) ) (not (and (on floor ?xxx) (in_motion ?xxx) (and (and (forall (?r - doggie_bed) (in_motion ?r pink_dodgeball) ) (on ?xxx ?xxx) (agent_holds ?xxx) ) (in_motion ?xxx) ) (not (agent_holds ?xxx ?xxx ?xxx) ) ) ) (in_motion ?xxx ?xxx) ) ) ) ) ) (agent_holds ?xxx) ) ) (in_motion ?xxx agent) (not (exists (?y - hexagonal_bin ?r - (either blue_cube_block ball)) (in_motion ?r ?r) ) ) (adjacent_side ?xxx ?xxx) (and (agent_holds ?xxx) (exists (?o - bridge_block ?j - watch) (in_motion agent ?j) ) (on ?xxx) ) ) (and (adjacent ?xxx) (< 2 (distance ?xxx room_center)) ) (on ?xxx bridge_block) ) (and (on ?xxx) (touch ?xxx door) ) ) (agent_holds ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (> (* (count-once preferenceC:blue_dodgeball) (count-nonoverlapping preferenceB:triangle_block) )
      (count-nonoverlapping preferenceB:pink)
    )
    3
  )
)
(:scoring
  (/
    5
    (* (count-once-per-external-objects preferenceB:dodgeball) 5 )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?l - cube_block)
    (game-conserved
      (in_motion ?l west_wall)
    )
  )
)
(:constraints
  (and
    (forall (?r - hexagonal_bin ?b - curved_wooden_ramp)
      (and
        (preference preferenceA
          (then
            (once (forall (?u - doggie_bed) (in_motion ?u ?u) ) )
            (once (agent_holds agent) )
            (hold (in_motion ?b ?b) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* 100 (+ (total-score) (+ 3 (* (+ (count-shortest preferenceA:hexagonal_bin) (count-overlapping preferenceA:triangle_block) )
              4
            )
          )
        )
      )
      (total-score)
    )
    3
  )
)
(:scoring
  (+ (count-once-per-objects preferenceA:pink_dodgeball:hexagonal_bin) 7 )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (exists (?t - hexagonal_bin)
      (game-conserved
        (on ?t ?t)
      )
    )
    (and
      (game-optional
        (in_motion agent)
      )
    )
  )
)
(:constraints
  (and
    (forall (?s ?r - color)
      (and
        (preference preferenceA
          (exists (?y ?h - triangular_ramp ?q - curved_wooden_ramp)
            (then
              (once (and (exists (?z - (either cube_block)) (and (not (agent_holds ?z) ) (and (exists (?n - block) (equal_z_position ?n) ) (not (toggled_on desk) ) (in_motion ?r ?q) (in_motion ?s ?q) (type ?z ?s) (on ?z) (and (not (and (not (same_type ?s) ) (and (and (not (>= 6 (distance ?q ?q)) ) (is_setup_object rug) (or (on ?q) (in ?s) ) (and (and (in_motion ?r pink) (adjacent_side ?s) ) (not (not (on ?z) ) ) ) (in_motion ?s) (object_orientation ?r) ) (and (in_motion ?q) (in ?z) ) ) ) ) (not (same_object ?s ?q) ) ) (not (and (not (agent_holds ?q) ) (agent_holds ?r) ) ) ) ) ) (< 1 7) ) )
              (hold (agent_holds ?r) )
              (once (agent_holds desk ?s) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (in_motion ?xxx ?xxx) )
        (hold (or (not (same_object ?xxx ?xxx) ) (not (in_motion ?xxx ?xxx) ) (< (distance room_center ?xxx) (distance desk ?xxx)) ) )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (<= 4 (count-nonoverlapping preferenceB:dodgeball:tan) )
)
(:scoring
  (count-once-per-objects preferenceB:blue_dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (not
      (and
        (> 2 (distance ?xxx back))
        (object_orientation ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (and (same_object ?xxx) (agent_holds ?xxx ?xxx) ) )
        (hold-while (on ?xxx ?xxx) (and (agent_holds ?xxx ?xxx) (on ?xxx) ) (in_motion ?xxx) )
        (hold (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once-per-objects preferenceA:dodgeball) (or 4 ) )
    (< (count-once-per-objects preferenceA:rug) (* (count-nonoverlapping preferenceA:dodgeball) (count-nonoverlapping preferenceA:green:blue_pyramid_block) )
    )
    (>= 7 (+ 50 8 )
    )
  )
)
(:scoring
  8
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (in_motion right)
  )
)
(:constraints
  (and
    (forall (?a - ball)
      (and
        (preference preferenceA
          (at-end
            (agent_holds )
          )
        )
        (preference preferenceB
          (exists (?k - pillow ?w ?b - dodgeball)
            (then
              (once (touch ?a) )
              (once (and (not (> (distance ?b ?b) 1) ) (not (not (agent_holds ?b ?b) ) ) ) )
              (once (agent_holds ?b ?w) )
            )
          )
        )
        (preference preferenceC
          (then
            (hold (in ?a ?a) )
            (once (not (rug_color_under ?a ?a) ) )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?x - chair ?m - building ?m - cube_block)
        (at-end
          (not
            (agent_holds ?m)
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* 4 20 (* (- 2 )
          (external-forall-maximize
            (count-once preferenceD:purple)
          )
          (external-forall-maximize
            (* (and (count-nonoverlapping preferenceB) ) (count-nonoverlapping preferenceA:pink) )
          )
          (count-once preferenceA:dodgeball)
          5
        )
      )
      (* (and (count-nonoverlapping preferenceD:purple) (* (count-longest preferenceB:pink_dodgeball) (count-nonoverlapping preferenceA:pink_dodgeball) )
          (+ (count-nonoverlapping preferenceC:blue_dodgeball:dodgeball) (- (total-time) )
            (= (- (count-nonoverlapping preferenceB:yellow) )
            )
          )
        )
        30
        4
        8
        (count-nonoverlapping preferenceB:yellow)
      )
    )
    (and
      (or
        (or
          (>= (external-forall-maximize 1 ) (+ (count-same-positions preferenceD:beachball:blue_pyramid_block:green) (count-once-per-objects preferenceD:dodgeball) (count-once-per-objects preferenceC:golfball) )
          )
          (or
            (>= (total-time) (count-once-per-objects preferenceD:beachball:basketball) )
          )
          (>= (count-once-per-objects preferenceC:dodgeball:yellow_pyramid_block) (count-nonoverlapping preferenceB:book) )
          (> (+ (count-once-per-objects preferenceB:dodgeball) (= (+ (count-overlapping preferenceB) (count-nonoverlapping preferenceD:beachball) )
                30
                (count-nonoverlapping preferenceA:golfball)
              )
            )
            (<= (<= 1 (count-nonoverlapping preferenceD:red) )
              (external-forall-maximize
                (count-nonoverlapping preferenceB:green:green)
              )
            )
          )
        )
        (< (count-nonoverlapping preferenceC:tan) 10 )
        (and
          (>= 5 (count-nonoverlapping preferenceC:alarm_clock) )
        )
      )
      (>= (count-once-per-objects preferenceB:yellow_cube_block) (* (count-nonoverlapping preferenceD:blue_pyramid_block) (count-nonoverlapping preferenceA:beachball) )
      )
    )
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (game-conserved
        (adjacent desk ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?e - building)
        (then
          (hold (forall (?u - cube_block ?v - hexagonal_bin ?d - (either golfball dodgeball)) (and (in_motion ?d) (agent_holds ?d ?d) ) ) )
          (once (touch pink_dodgeball ?e) )
          (once (in_motion ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (= 18 3 )
)
(:scoring
  1
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?x ?s - dodgeball)
    (game-conserved
      (on ?s)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?m - cube_block)
        (exists (?r - hexagonal_bin ?s - block)
          (then
            (once (and (agent_holds ?s) (not (and (in ?s ?s) (on desk ?s) (same_color ?m) (agent_holds ?m ?m) ) ) ) )
            (once (not (not (and (on ?m) (and (and (not (< 1 (distance ?s desk)) ) (on ?s ?s) ) (not (game_start desk) ) ) ) ) ) )
            (once (not (in_motion ?s ?m) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (> (* 100 (+ (* (count-nonoverlapping preferenceA:block) (count-once-per-objects preferenceA:doggie_bed) )
        (* 50 (count-nonoverlapping preferenceA:pink) )
      )
    )
    0.5
  )
)
(:scoring
  (* 10 15 (count-once-per-objects preferenceA:pink) (count-once-per-objects preferenceA:pyramid_block) (- (* 4 (* 3 (+ (total-time) (* (not 40 ) (count-once-per-objects preferenceA:top_drawer:orange) )
          )
          (* 1 5 )
        )
        5
      )
    )
    3
    (total-score)
    (count-nonoverlapping preferenceA:golfball)
    (* (= (count-nonoverlapping preferenceA:pink_dodgeball:dodgeball) (count-nonoverlapping preferenceA:pink) )
      (count-nonoverlapping preferenceA:pink)
      (count-once preferenceA:beachball)
      10
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?k - dodgeball)
    (game-conserved
      (agent_holds ?k ?k)
    )
  )
)
(:constraints
  (and
    (forall (?j ?k - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?l - ball)
            (then
              (once (and (on pillow ?l) (touch ?j rug) ) )
              (once (agent_holds ?l) )
            )
          )
        )
      )
    )
    (forall (?p - tall_cylindrical_block)
      (and
        (preference preferenceB
          (exists (?i - (either dodgeball golfball) ?k - cube_block)
            (then
              (hold (agent_holds ?k) )
              (once (agent_holds ?k) )
              (hold (on ?k) )
              (hold (on ?p) )
            )
          )
        )
      )
    )
    (preference preferenceC
      (then
        (once (in_motion rug) )
        (hold-while (= 0.5 1) (adjacent pink) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (= (+ 2 (- 3 )
    )
    (count-total preferenceA:yellow)
  )
)
(:scoring
  (count-shortest preferenceB:dodgeball:green)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (exists (?v - wall ?b - ball)
        (exists (?q ?w - hexagonal_bin ?e - (either tall_cylindrical_block book golfball blue_cube_block dodgeball floor key_chain))
          (game-conserved
            (not
              (in_motion bed ?b)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?v - game_object ?e - shelf)
        (exists (?o - cube_block)
          (exists (?x - ball)
            (at-end
              (in_motion ?o ?x)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= 18 (+ (- (total-time) )
      (+ (* (count-nonoverlapping preferenceA:beachball:golfball) (* 0 )
        )
        (count-once preferenceA:blue_dodgeball)
      )
    )
  )
)
(:scoring
  (+ (- (external-forall-maximize 4 ) )
    (count-nonoverlapping preferenceA)
    15
    (* (count-nonoverlapping preferenceA:beachball) (count-once-per-objects preferenceA:green) (* 6 100 )
    )
    (total-time)
    (count-once-per-objects preferenceA:red:green)
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?q - hexagonal_bin ?i - triangular_ramp)
    (forall (?j - triangular_ramp ?h - (either rug rug))
      (exists (?t - (either yellow yellow))
        (game-conserved
          (agent_holds ?t ?i)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold (not (and (adjacent bed ?xxx) (not (agent_holds agent ?xxx ?xxx) ) ) ) )
        (once (on rug) )
        (once (on ?xxx) )
      )
    )
    (forall (?p - (either cube_block golfball cylindrical_block game_object blue_cube_block dodgeball blue_cube_block))
      (and
        (preference preferenceB
          (then
            (once (agent_holds ?p ?p) )
            (hold (agent_holds ?p) )
            (hold (and (in_motion floor) (rug_color_under bed ?p) ) )
          )
        )
        (preference preferenceC
          (exists (?w - game_object)
            (exists (?l - (either pillow pyramid_block))
              (exists (?x - beachball)
                (exists (?u - building)
                  (then
                    (once (not (not (and (on ?p ?l) ) ) ) )
                    (once (and (agent_holds ?x) (in_motion ?u) ) )
                    (once (not (not (not (> (distance desk 0 ?p) (distance ?u green_golfball)) ) ) ) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceD
      (then
        (once (in_motion ?xxx ?xxx) )
        (hold (faces ?xxx) )
        (once (touch ?xxx) )
      )
    )
    (preference preferenceE
      (exists (?f - sliding_door)
        (then
          (hold (in_motion ?f) )
          (once (adjacent ?f ?f) )
          (once (and (and (same_color ?f south_west_corner) (is_setup_object ?f ?f) (= (distance room_center ?f) (distance ?f) 1 2) ) (agent_holds ?f) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (* (count-once-per-objects preferenceA:pink) (count-nonoverlapping preferenceE:purple:yellow) )
        (count-once-per-objects preferenceB:beachball)
      )
      (count-nonoverlapping preferenceC:yellow)
    )
    (<= (count-once-per-objects preferenceE:triangle_block) (* (* (- (+ 9 (- 0 )
            )
          )
          (< (count-nonoverlapping preferenceC:dodgeball) 3 )
        )
        (- (count-nonoverlapping preferenceB:brown:beachball) )
      )
    )
    (>= (* 2 )
      5
    )
  )
)
(:scoring
  (* (>= 3 (count-once-per-objects preferenceA:pink:red) )
    (count-once preferenceA:basketball:pink)
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?k - game_object)
    (not
      (and
        (and
          (game-conserved
            (in ?k)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - dodgeball)
      (and
        (preference preferenceA
          (exists (?y - game_object)
            (then
              (hold-while (is_setup_object ?y) (not (not (< (distance room_center ?a) (distance front_left_corner agent)) ) ) )
              (once (in_motion ?a) )
              (once (not (object_orientation floor) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (< (count-nonoverlapping preferenceA:yellow) (* (* 15 (count-nonoverlapping preferenceA:doggie_bed) (external-forall-maximize (count-once-per-external-objects preferenceA:pink) ) )
        (count-once-per-objects preferenceA:dodgeball)
        (count-nonoverlapping preferenceA:alarm_clock)
      )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-optional
      (in_motion desk pink)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b - triangular_ramp)
        (exists (?j - hexagonal_bin)
          (then
            (once (and (touch ?j) (and (not (in_motion ?j floor) ) (in ?b agent) (and (and (agent_holds ?b ?j) (in_motion ?b) ) (in_motion ?b) (and (in_motion ?b) (not (agent_holds ?j) ) (on ?j ?j) (and (faces ?b ?j) (and (not (not (on ?b) ) ) (not (on ?j ?b) ) ) ) (in_motion ?b) ) (exists (?r - color) (in ?j) ) ) ) (and (agent_holds ?b ?b) (in ?b ?b) ) (on ?j) ) )
          )
        )
      )
    )
    (preference preferenceB
      (at-end
        (and
          (not
            (= 1 1 1)
          )
          (and
            (agent_holds ?xxx rug)
            (and
              (not
                (in_motion rug)
              )
              (in_motion ?xxx)
              (adjacent ?xxx)
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?i - teddy_bear)
        (exists (?t - cube_block)
          (then
            (hold (not (not (in ?i) ) ) )
            (hold-while (forall (?k - hexagonal_bin) (in_motion ?t ?t) ) (not (in_motion ?i) ) (and (on ?t) (and (or (and (not (in_motion ?i agent) ) (and (and (not (agent_holds ?t) ) (in_motion front) ) (not (agent_holds ?t) ) ) ) (on ?i ?t ?i) ) (in_motion ?i) ) ) )
            (hold (not (and (agent_holds ?i) (and (and (agent_holds ?i ?t) (not (in_motion ?i) ) ) (and (and (not (in_motion ?t floor) ) (not (in_motion ?t ?t) ) ) (on ?i) ) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 6 (* (+ 50 (count-nonoverlapping preferenceB:dodgeball) )
      (total-score)
      2
    )
  )
)
(:scoring
  (count-once preferenceA:yellow)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (exists (?u - teddy_bear ?r - color)
        (exists (?n - (either pyramid_block pink pyramid_block wall wall flat_block ball) ?l - ball)
          (exists (?s - hexagonal_bin ?g - hexagonal_bin)
            (game-conserved
              (in_motion ?r ?r)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?q ?t - ball ?v - (either mug cellphone golfball) ?i - hexagonal_bin)
        (then
          (hold (agent_holds ?i) )
          (once (and (and (forall (?s ?f - rug) (not (in ?s desk) ) ) (on bed) ) (in_motion desk ?i) ) )
          (hold (touch ?i) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 3 (count-once-per-objects preferenceA:pink) )
    (>= (count-nonoverlapping preferenceA:golfball) (count-nonoverlapping preferenceA:pink) )
    (or
      (not
        (>= 1 (count-once preferenceA:top_drawer:hexagonal_bin) )
      )
      (>= 4 (* (count-nonoverlapping preferenceA:triangle_block) (count-nonoverlapping preferenceA:dodgeball:basketball) )
      )
    )
  )
)
(:scoring
  (count-nonoverlapping preferenceA:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (not
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (not (in_motion floor) ) )
        (once (not (in ?xxx) ) )
        (once (touch ?xxx color) )
      )
    )
    (forall (?c - block)
      (and
        (preference preferenceB
          (exists (?j - color)
            (exists (?b - ball)
              (exists (?x - shelf)
                (then
                  (once (on ?x) )
                  (once (and (and (not (not (and (and (adjacent ?j) (in_motion ) ) (exists (?o - ball) (in_motion ?j agent) ) ) ) ) (not (exists (?l - dodgeball) (and (and (faces bed) (agent_holds ?x bed) ) (agent_holds ?l door) (agent_holds ?j) ) ) ) ) (on ?j) ) )
                  (hold (in_motion desk) )
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?g - hexagonal_bin ?o - doggie_bed)
        (exists (?x - dodgeball ?s - chair)
          (then
            (once (not (and (same_type ?s upside_down) (not (in_motion upright) ) ) ) )
            (once (same_color ?o ?o) )
            (hold (in_motion ?o) )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?l - hexagonal_bin)
        (then
          (once (agent_holds ?l ?l) )
          (once (in_motion ?l) )
          (once (in_motion ?l) )
        )
      )
    )
    (preference preferenceE
      (exists (?x - cylindrical_block ?j - dodgeball)
        (at-end
          (not
            (on ?j)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-nonoverlapping preferenceA:golfball:dodgeball) (not (+ (count-same-positions preferenceB) (count-unique-positions preferenceC:block:doggie_bed) )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (not
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold (agent_holds ?xxx ?xxx) )
        (hold (not (touch desk ?xxx) ) )
        (once (on ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preferenceA:basketball) (count-nonoverlapping preferenceA:basketball) )
)
(:scoring
  (count-once preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?q - desk_shelf)
    (game-optional
      (on bed agent)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?j - hexagonal_bin)
        (exists (?m - game_object ?f - (either cube_block cube_block) ?f - hexagonal_bin ?o - triangular_ramp)
          (exists (?i - color ?g - hexagonal_bin)
            (then
              (once (not (object_orientation ?j ?g) ) )
              (once (in ?j) )
              (once (not (equal_z_position ?o) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?r - wall)
        (exists (?k - dodgeball)
          (then
            (hold (on ?k ?r) )
            (once (and (and (in ?r) (not (in_motion ?r) ) ) (in_motion ?r ?k) ) )
            (once (<= 1 (distance side_table 10)) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count-once-per-objects preferenceB:dodgeball) (>= (+ (* (- (+ (* (* (* (external-forall-maximize 3 ) (* 1 6 )
                    )
                    5
                  )
                  (* (* (count-nonoverlapping preferenceA:pink) (* (count-shortest preferenceB) (* (count-once-per-objects preferenceA:alarm_clock) (external-forall-maximize (count-once-per-objects preferenceA:basketball) ) )
                      )
                    )
                    (count-nonoverlapping preferenceA:hexagonal_bin)
                  )
                )
                (* 10 10 )
              )
            )
            (* (* (count-once-per-objects preferenceB:red:doggie_bed) 2 )
              6
              (= (+ (count-once-per-objects preferenceB:dodgeball) (count-nonoverlapping preferenceB:golfball) )
                (+ 8 7 )
                (count-once-per-objects preferenceB:red)
              )
            )
            (count-nonoverlapping preferenceA:golfball)
          )
          (<= (count-once-per-objects preferenceB:hexagonal_bin) (count-nonoverlapping preferenceA:tan:beachball) )
        )
        (- 7 )
      )
    )
    (external-forall-minimize
      (count-nonoverlapping preferenceA:beachball)
    )
  )
)
(:scoring
  (count-once-per-objects preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (and
      (in agent ?xxx)
      (adjacent ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?q - block ?q - pyramid_block)
        (then
          (hold-while (on ?q) (in_motion bed) )
          (hold (not (not (in_motion desk) ) ) )
          (once (or (not (touch ?q) ) (on ?q) (and (adjacent ?q) (and (above desk) (not (or (not (and (agent_holds ?q) (and (agent_holds ?q ?q) (and (or (not (and (on ?q ?q) (and (agent_holds ?q ?q) (and (and (or (on ?q ?q) (touch ?q) ) (> 2 1) ) (not (in_motion ?q ?q) ) ) (agent_holds ?q) ) ) ) (in_motion ?q) ) (not (agent_holds ?q ?q) ) (not (faces ?q) ) ) ) ) ) (not (agent_holds ?q ?q) ) ) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (count-nonoverlapping preferenceA:beachball) 0 )
    (and
      (>= (+ (* (* (- (* 10 )
              )
              (count-total preferenceA:pyramid_block:dodgeball)
            )
            5
          )
          50
        )
        (total-score)
      )
    )
    (= (count-nonoverlapping preferenceA:golfball) 3 )
  )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (forall (?f - beachball ?d - teddy_bear)
      (and
        (preference preferenceA
          (exists (?k - dodgeball)
            (exists (?n - teddy_bear)
              (forall (?w - (either floor pencil dodgeball))
                (then
                  (once-measure (adjacent ?k ?n) (distance ?w 7) )
                  (once (and (agent_holds ?w) (and (on ?n) (not (and (agent_holds ?k) (agent_holds ?w) ) ) ) ) )
                  (once (not (not (on top_drawer ?w) ) ) )
                )
              )
            )
          )
        )
        (preference preferenceB
          (then
            (hold-while (in_motion ?d ?d) (on ?d ?d) )
            (once-measure (agent_holds ?d desk) (distance ?d desk) )
            (once (agent_holds ?d ?d) )
          )
        )
      )
    )
    (preference preferenceC
      (at-end
        (same_color ?xxx)
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preferenceC:dodgeball) (* 25 4 )
  )
)
(:scoring
  (* (+ (= (count-longest preferenceA:golfball) (count-once-per-objects preferenceB:dodgeball:yellow_pyramid_block) )
      (* (* 5 (external-forall-maximize (external-forall-maximize 0.7 ) ) )
        (count-nonoverlapping preferenceB:brown)
      )
    )
    7
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?u - golfball)
    (game-optional
      (in_motion ?u ?u)
    )
  )
)
(:constraints
  (and
    (forall (?i - color)
      (and
        (preference preferenceA
          (exists (?q - (either tall_cylindrical_block alarm_clock))
            (then
              (hold (not (not (not (touch ?q agent) ) ) ) )
              (once (agent_holds ?q ?i) )
              (once (agent_holds ?i ?i) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count-nonoverlapping preferenceA:basketball:orange) 20 )
)
(:scoring
  2
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-optional
      (and
        (adjacent ?xxx)
        (and
          (= 8 (distance_side ?xxx ?xxx) (distance room_center agent))
          (same_color ?xxx)
        )
      )
    )
    (game-conserved
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold-while (agent_holds ?xxx ?xxx) (not (in ?xxx) ) (agent_holds ?xxx) )
        (hold (not (on ?xxx) ) )
        (once (adjacent_side ?xxx ?xxx) )
      )
    )
    (preference preferenceB
      (then
        (once (< 8 1) )
        (once (not (agent_holds front) ) )
        (once (in ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (= (count-once-per-objects preferenceB:dodgeball) (count-once-per-objects preferenceA:blue_dodgeball:dodgeball) )
)
(:scoring
  (* (count-nonoverlapping preferenceA:beachball:red:green) (* (count-overlapping preferenceB:cube_block) (count-once preferenceA:basketball) )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?h - ball)
    (and
      (exists (?e - hexagonal_bin)
        (and
          (game-conserved
            (on ?h ?e)
          )
          (game-conserved
            (faces ?e)
          )
          (and
            (and
              (or
                (game-conserved
                  (adjacent ?e)
                )
                (game-conserved
                  (in_motion agent)
                )
                (game-conserved
                  (on ?h ?h)
                )
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold (and (not (not (agent_holds ?xxx) ) ) (and (not (between ?xxx) ) (in ?xxx) ) ) )
        (hold-while (> 1 1) (agent_holds ?xxx) )
        (hold-while (not (in ?xxx ?xxx) ) (faces agent) )
      )
    )
  )
)
(:terminal
  (= (- (* (count-nonoverlapping preferenceA:basketball) (count-longest preferenceA:doggie_bed) (count-once-per-objects preferenceA:dodgeball) (- (* 2 15 )
        )
        (+ (= (* (count-once-per-objects preferenceA:dodgeball:hexagonal_bin) (count-longest preferenceA:beachball) )
            (+ 5 (* 10 (* (* (* (external-forall-maximize (- 10 )
                      )
                      5
                      2
                    )
                    (+ 5 )
                  )
                  (count-once-per-objects preferenceA:golfball:dodgeball)
                )
              )
              (count-unique-positions preferenceA:yellow)
              (count-nonoverlapping preferenceA:alarm_clock)
              (* 6 1 )
              (* (* (+ (count-nonoverlapping preferenceA:top_drawer) )
                  (* 2 )
                )
                (total-score)
              )
              (count-nonoverlapping preferenceA:basketball)
            )
            (count-once-per-objects preferenceA:beachball:dodgeball)
          )
          (* (* (= 1 6 )
              (+ 5 (* (+ (count-once-per-objects preferenceA:dodgeball) (count-once-per-objects preferenceA:beachball:golfball) (count-nonoverlapping preferenceA:hexagonal_bin:green) (count-nonoverlapping preferenceA:basketball:alarm_clock) (count-unique-positions preferenceA) (count-nonoverlapping preferenceA:dodgeball) )
                  5
                )
              )
            )
            (count-once-per-objects preferenceA:beachball)
            (count-nonoverlapping preferenceA:dodgeball)
          )
        )
        3
        (count-nonoverlapping preferenceA:hexagonal_bin)
        (count-increasing-measure preferenceA:yellow_pyramid_block)
        (+ (* (count-nonoverlapping preferenceA:beachball) (total-score) )
          (count-once-per-objects preferenceA:book)
          3
          (count-nonoverlapping preferenceA:basketball)
          3
          (count-once preferenceA:golfball:bed)
        )
      )
    )
    (= 2 (count-once-per-objects preferenceA:blue_pyramid_block) )
  )
)
(:scoring
  (= (* (* 5 10 )
      (count-nonoverlapping preferenceA:red:dodgeball)
    )
    (count-once-per-objects preferenceA:beachball)
    (+ (external-forall-maximize (* (count-nonoverlapping preferenceA:pink_dodgeball:basketball) 1 )
      )
      (count-nonoverlapping preferenceA:beachball)
      (* 8 10 (or (* (total-time) (* (* 100 (- (= (* (count-nonoverlapping preferenceA:dodgeball) (count-nonoverlapping preferenceA:golfball:book) )
                    (external-forall-minimize
                      (+ (count-shortest preferenceA:dodgeball) (* (count-nonoverlapping preferenceA:yellow) 2 (+ (* (count-shortest preferenceA:basketball) (external-forall-maximize (count-nonoverlapping preferenceA:pink) ) )
                            (count-nonoverlapping preferenceA:basketball:red_pyramid_block:hexagonal_bin)
                          )
                        )
                        5
                        (count-nonoverlapping-measure preferenceA:basketball:doggie_bed)
                        (* (count-nonoverlapping preferenceA:red_pyramid_block:basketball) (count-nonoverlapping preferenceA:purple:bed:dodgeball) )
                        (+ 4 (* (- (count-longest preferenceA:dodgeball) (count-nonoverlapping preferenceA:red) ) (* (count-increasing-measure preferenceA:dodgeball) 1 )
                          )
                        )
                      )
                    )
                    (count-once preferenceA:pyramid_block:triangle_block)
                  )
                )
              )
              (* (* (* (count-overlapping preferenceA:beachball) (* (count-nonoverlapping preferenceA:brown) (count-once-per-objects preferenceA:dodgeball) )
                    (/
                      3
                      (+ 2 (count-nonoverlapping preferenceA:purple) )
                    )
                  )
                  (count-nonoverlapping-measure preferenceA:dodgeball:blue_dodgeball)
                  (+ (* 1 (count-once-per-objects preferenceA:basketball:white) (count-longest preferenceA:basketball) )
                    3
                  )
                )
                3
              )
            )
          )
          (count-once preferenceA:pink:hexagonal_bin:beachball)
          (* (* (+ 6 (- (* 4 (count-nonoverlapping preferenceA:beachball) 4 (+ (total-time) (+ (count-once preferenceA:dodgeball) 5 )
                    )
                    (count-nonoverlapping preferenceA:red)
                    2
                  )
                )
              )
              (* (count-nonoverlapping preferenceA:dodgeball:beachball) (+ (count-increasing-measure preferenceA:dodgeball) (count-nonoverlapping preferenceA:beachball) (count-once-per-objects preferenceA:yellow) (count-nonoverlapping preferenceA:orange) (count-nonoverlapping preferenceA:dodgeball:beachball) )
              )
            )
            (count-same-positions preferenceA:side_table)
          )
        )
        (count-overlapping preferenceA:triangle_block:doggie_bed)
        (+ 7 (* 30 (* (total-score) (+ (count-once-per-objects preferenceA:basketball) 0 )
              (* (count-nonoverlapping preferenceA:dodgeball) 5 )
            )
          )
        )
      )
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?q - (either main_light_switch cube_block))
    (forall (?z - hexagonal_bin)
      (game-optional
        (not
          (not
            (in ?z)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?w - block)
        (at-end
          (not
            (not
              (and
                (agent_holds ?w ?w)
                (>= (distance front) 2)
                (in ?w)
                (not
                  (not
                    (in_motion ?w ?w)
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 40 (* (count-nonoverlapping preferenceA:basketball:golfball) )
  )
)
(:scoring
  (count-once preferenceA:dodgeball:block)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (not
      (agent_holds ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (not (on ?xxx pillow) ) )
        (hold (< 1 (distance ?xxx ?xxx)) )
        (hold (and (is_setup_object ?xxx ?xxx) (not (in_motion ?xxx) ) (< (distance room_center ?xxx 10) (building_size 8 10)) ) )
      )
    )
    (preference preferenceB
      (then
        (once (in_motion ?xxx) )
        (once (and (agent_holds ?xxx) (in_motion ?xxx ?xxx) ) )
        (once (not (not (in_motion ?xxx ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (>= (* 10 (count-overlapping preferenceB:beachball) )
    3
  )
)
(:scoring
  (* (count-shortest preferenceB:beachball) (+ (* (count-once-per-objects preferenceB:pink:beachball) 7 )
      5
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (not
      (not
        (not
          (agent_holds ?xxx ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?c - (either basketball dodgeball))
      (and
        (preference preferenceA
          (then
            (once (not (not (and (in ?c ?c) (and (and (agent_holds ?c ?c) (and (and (in_motion ?c) (in_motion ?c brown) ) (in_motion ?c) ) ) (agent_holds ?c) ) ) ) ) )
            (once (not (in_motion ?c ?c) ) )
            (once (agent_holds ?c) )
          )
        )
      )
    )
    (forall (?z - shelf ?u ?s - cylindrical_block)
      (and
        (preference preferenceB
          (then
            (forall-sequence (?o - game_object)
              (then
                (once (in ?o) )
                (once (and (and (not (in_motion ?s) ) (touch ?o) ) (in ?s) ) )
                (hold (not (in south_west_corner) ) )
              )
            )
            (once (or (agent_holds desk ?u) (not (exists (?f - teddy_bear ?j - building) (agent_holds ?s desktop) ) ) (adjacent ?u) (not (agent_holds ?s) ) ) )
            (hold (agent_holds ?u) )
          )
        )
        (preference preferenceC
          (then
            (once (exists (?c - game_object) (agent_holds ?u ?c) ) )
            (hold (agent_holds ?u ?s) )
            (once (agent_holds ?s) )
          )
        )
        (preference preferenceD
          (then
            (once (same_color ?s floor) )
            (once (object_orientation ?u) )
            (once (agent_holds ?s) )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (and
      (>= 5 7 )
    )
  )
)
(:scoring
  1
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (in ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?v - game_object)
        (then
          (hold (not (and (agent_holds ?v ?v) (touch ?v ?v) ) ) )
          (once (agent_holds ?v ?v) )
          (once (and (in ?v ?v) (in_motion ?v) (on ?v ?v) ) )
          (hold (and (same_color ?v ?v) (in ?v) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* (* (* 2 (count-once-per-objects preferenceA:pink) )
          (count-nonoverlapping preferenceA:yellow)
        )
        (* 0 (count-nonoverlapping preferenceA:basketball) (total-score) )
      )
      3
    )
    (count-once-per-external-objects preferenceA:beachball)
  )
)
(:scoring
  (count-nonoverlapping preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (forall (?p - pyramid_block)
      (in_motion ?p agent)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (at-end
        (and
          (touch ?xxx ?xxx)
          (and
            (and
              (in_motion ?xxx desk)
              (agent_holds ?xxx)
            )
            (and
              (not
                (in_motion ?xxx ?xxx)
              )
              (and
                (touch ?xxx)
                (forall (?w - drawer)
                  (agent_holds ?w)
                )
              )
            )
          )
          (and
            (in_motion ?xxx ball)
            (and
              (and
                (and
                  (agent_holds ?xxx ?xxx)
                  (not
                    (not
                      (or
                        (< 1 (distance room_center 1))
                        (agent_holds ?xxx ?xxx)
                      )
                    )
                  )
                )
                (agent_holds ?xxx ?xxx)
                (and
                  (on ?xxx)
                  (touch ?xxx floor)
                )
                (on agent)
              )
              (adjacent bed ?xxx)
            )
          )
          (exists (?q - dodgeball)
            (in_motion ?q ?q)
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-nonoverlapping preferenceA:book) (> (+ (count-nonoverlapping preferenceA:pink_dodgeball) 30 )
        (* (count-once-per-objects preferenceA:hexagonal_bin) (* 300 (total-score) )
        )
      )
    )
    (< (= (count-nonoverlapping preferenceA:dodgeball) 3 (= (* (count-nonoverlapping preferenceA:dodgeball:basketball) 5 )
          10
        )
      )
      (count-once preferenceA:top_drawer)
    )
  )
)
(:scoring
  (count-nonoverlapping preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?n - hexagonal_bin ?e - hexagonal_bin)
    (and
      (forall (?c - hexagonal_bin)
        (exists (?v - (either game_object dodgeball))
          (game-conserved
            (not
              (faces ?c)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?n - hexagonal_bin ?s - game_object)
      (and
        (preference preferenceA
          (then
            (hold (not (not (agent_holds ?s ?s) ) ) )
            (once (not (and (and (in_motion ?s ?s) (not (not (type ?s ?s) ) ) ) (in_motion front ?s) ) ) )
            (hold-while (agent_holds ?s agent) (in_motion ?s) )
          )
        )
        (preference preferenceB
          (exists (?e - game_object)
            (then
              (hold-while (agent_holds main_light_switch) (not (agent_holds ?e ?e) ) )
              (once (in_motion ?s ?s) )
              (hold (in_motion ?e) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count-nonoverlapping preferenceA:pink) 180 )
    (>= (count-unique-positions preferenceA:blue_dodgeball) (* (count-nonoverlapping preferenceB:basketball) (count-nonoverlapping preferenceB:dodgeball) (* 15 (+ 0 (count-increasing-measure preferenceB:dodgeball) 3 (- 20 )
            (* (external-forall-maximize (+ (* (* (* (* (count-once-per-objects preferenceA:pink_dodgeball) (count-nonoverlapping preferenceB:blue_dodgeball) )
                        30
                      )
                      (* (* (count-nonoverlapping preferenceB:red) (* (count-nonoverlapping preferenceB:basketball) (* 100 (count-nonoverlapping preferenceA:yellow_pyramid_block) )
                            3
                            2
                          )
                        )
                        (+ (count-nonoverlapping preferenceA:beachball) (count-nonoverlapping preferenceB:pink_dodgeball:red_pyramid_block) )
                        10
                      )
                    )
                    5
                  )
                  (+ (count-total preferenceB:hexagonal_bin) )
                )
              )
              (count-total preferenceA:yellow:yellow)
            )
            10
          )
        )
      )
    )
  )
)
(:scoring
  (total-score)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (and
      (exists (?p ?q ?h - cube_block ?b - hexagonal_bin)
        (forall (?k ?f - ball)
          (game-conserved
            (and
              (open ?b ?k)
              (broken floor)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (hold-to-end (not (faces ?xxx) ) )
        (any)
        (hold (and (not (not (and (and (on ?xxx) (in ?xxx ?xxx) (agent_holds door ?xxx) ) (in ?xxx ?xxx) ) ) ) (rug_color_under ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-nonoverlapping preferenceA:dodgeball) 8 )
    (> (count-once-per-objects preferenceA:pink:blue_pyramid_block) (* (count-nonoverlapping preferenceA:dodgeball) (count-nonoverlapping preferenceA:white) )
    )
    (>= (- 5 )
      100
    )
  )
)
(:scoring
  (* (* (count-nonoverlapping preferenceA:orange) (count-nonoverlapping preferenceA:pink) )
    (total-score)
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?y - dodgeball)
    (game-optional
      (> (distance ?y) (distance ?y desk))
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (at-end
        (agent_holds ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (<= (* (count-shortest preferenceA:beachball) (count-nonoverlapping preferenceA:dodgeball) 0 (count-once-per-external-objects preferenceA:yellow) (count-nonoverlapping preferenceA:dodgeball:dodgeball) 2 )
      (count-once-per-objects preferenceA:pink)
    )
    (= 10 (count-unique-positions preferenceA:dodgeball:basketball) )
  )
)
(:scoring
  (* (count-nonoverlapping preferenceA:dodgeball:hexagonal_bin) (count-nonoverlapping preferenceA:book) (count-increasing-measure preferenceA:pyramid_block) )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?n - (either basketball ball book cylindrical_block key_chain cellphone pen) ?p - (either laptop laptop) ?s - dodgeball)
    (game-optional
      (agent_holds ?s)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?g - ball)
        (exists (?m - hexagonal_bin)
          (then
            (hold (and (and (broken ?g ?g) (on ?m) ) ) )
            (once (touch ?m) )
            (once (and (not (not (on ?g ?g) ) ) (and (not (in_motion ?g ?g) ) (and (and (and (exists (?y - game_object ?s - (either basketball basketball)) (and (not (and (on ?g ?m) (agent_holds ?s ?g) (in_motion ?g ?s) ) ) (in_motion ?m top_shelf) ) ) (not (same_object ?g ?g) ) (agent_holds pillow) ) (on ?m ?g) ) (in ?m upright) ) (adjacent ?m) ) (not (and (not (and (not (in ?g ?g) ) (and (in_motion ?m) (< 4 1) ) ) ) (and (adjacent ?g ?m) (in_motion ?m) ) ) ) ) )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?x ?c - hexagonal_bin)
        (then
          (once (agent_holds rug ?x) )
          (hold (= 1 (distance ?x ?x) (distance_side ?x)) )
          (hold-while (and (< 2 (distance room_center agent)) (not (not (in_motion ?x ?x) ) ) ) (in_motion ?x ?c ?x) (touch ?x) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-nonoverlapping preferenceB:bed:dodgeball) 3 )
    (>= 8 (= 3 (* (+ 6 10 (count-once-per-external-objects preferenceB:pink) )
          (* 10 (= (total-time) )
          )
        )
      )
    )
    (>= (+ (count-nonoverlapping preferenceB:pink:beachball) 5 )
      (* 5 (* 3 40 )
      )
    )
  )
)
(:scoring
  (count-once-per-objects preferenceB:book:pink_dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?u - chair)
    (game-conserved
      (not
        (not
          (in_motion ?u)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?d - game_object)
        (then
          (once (in_motion ?d) )
          (hold (in_motion agent) )
          (once (type ?d) )
        )
      )
    )
    (preference preferenceB
      (at-end
        (adjacent ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once-per-objects preferenceA:dodgeball:pink) 3 )
    (< (count-once-per-objects preferenceB:pink_dodgeball) 30 )
  )
)
(:scoring
  (+ (count-nonoverlapping preferenceB:beachball) (* (count-nonoverlapping preferenceB:doggie_bed) )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?e - hexagonal_bin)
    (game-conserved
      (and
        (< (distance front 8 ?e) (distance ?e room_center))
        (not
          (< (distance ?e ?e) 5)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?g - hexagonal_bin)
        (then
          (once (and (on ?g door) (and (same_color ?g) (in_motion desk) (and (in_motion agent) (not (adjacent_side ?g ?g) ) ) ) ) )
          (once (not (in ?g) ) )
          (hold-while (exists (?k - curved_wooden_ramp ?i ?d - (either chair cube_block book red)) (and (touch ?g) (and (and (in_motion ?i) (and (agent_holds ?d ?d) (in ?g) ) ) (agent_holds ?g ?i) ) (not (agent_holds ?i) ) ) ) (in ?g ?g) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 5 )
)
(:scoring
  (* 100 2 )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (in_motion agent)
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b - (either blue_cube_block pyramid_block))
        (exists (?a - curved_wooden_ramp)
          (then
            (once (in_motion tan) )
            (once (not (agent_holds ?a ?b) ) )
            (once (or (agent_holds ?b ?a) (not (and (and (and (not (agent_holds ?b ?a) ) (not (and (agent_holds ?a ?b) (on ?b ?b) ) ) ) (not (adjacent ?a) ) ) (in_motion ?b ?b) ) ) (not (and (> 5 1) (and (agent_holds ?a ?a) (and (and (agent_holds bed ?b) (not (in_motion ?b ?a) ) (in_motion ) ) (exists (?i - curved_wooden_ramp ?t - (either dodgeball cylindrical_block)) (not (on ?b) ) ) ) ) (agent_holds ?a bed) ) ) (or (and (not (in agent ?b) ) (on ?a ?a) ) (agent_holds ?b ?a) ) ) )
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (not (agent_holds ?xxx) ) )
        (once (in_motion ?xxx) )
        (once (and (and (agent_holds ?xxx ?xxx) (not (not (not (not (not (forall (?i - (either teddy_bear golfball)) (adjacent ?i) ) ) ) ) ) ) ) (on ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (or
    (>= 18 (count-nonoverlapping preferenceB:dodgeball) )
    (> (count-overlapping preferenceA:dodgeball) (count-once preferenceA:purple:green) )
  )
)
(:scoring
  (count-nonoverlapping preferenceA:red:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?n - hexagonal_bin)
    (forall (?d - (either bridge_block mug basketball) ?p - dodgeball)
      (and
        (exists (?g - pyramid_block ?v - doggie_bed)
          (and
            (and
              (exists (?j - hexagonal_bin)
                (forall (?q - hexagonal_bin ?b - beachball)
                  (game-conserved
                    (not
                      (agent_holds brown)
                    )
                  )
                )
              )
              (game-conserved
                (or
                  (in_motion ?n ?n)
                )
              )
            )
            (and
              (or
                (and
                  (exists (?l - hexagonal_bin)
                    (forall (?d - book ?z - teddy_bear)
                      (and
                        (exists (?k - building)
                          (and
                            (game-optional
                              (exists (?f - cube_block)
                                (in_motion ?l ?z)
                              )
                            )
                            (forall (?e - golfball)
                              (exists (?s ?h - dodgeball ?h - (either desktop))
                                (game-conserved
                                  (exists (?i - hexagonal_bin ?s - hexagonal_bin ?s - (either dodgeball doggie_bed) ?u - cube_block)
                                    (in_motion ?e ?l)
                                  )
                                )
                              )
                            )
                            (game-conserved
                              (in_motion block)
                            )
                          )
                        )
                        (game-optional
                          (not
                            (in_motion ?n ?l)
                          )
                        )
                        (and
                          (exists (?g - cube_block ?r - chair)
                            (exists (?k - hexagonal_bin ?x ?w - (either triangle_block))
                              (exists (?f - building)
                                (exists (?s - hexagonal_bin)
                                  (forall (?e - game_object)
                                    (game-optional
                                      (in_motion ?l)
                                    )
                                  )
                                )
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                  (and
                    (game-optional
                      (and
                        (agent_holds ?p)
                        (on desk)
                        (not
                          (not
                            (not
                              (not
                                (and
                                  (agent_holds ?v)
                                  (agent_holds ?v)
                                )
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (game-conserved
          (agent_holds ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?c - (either golfball blue_cube_block) ?j - dodgeball)
      (and
        (preference preferenceA
          (exists (?o - cube_block ?d - game_object)
            (exists (?h - hexagonal_bin ?i - ball ?h - hexagonal_bin)
              (then
                (once (or (and (adjacent ?j) (toggled_on ?h agent) ) (and (in ?j ?d) (< 7 (distance 3 ?d)) ) ) )
                (once (not (and (not (in_motion ?j) ) (not (not (and (agent_holds front) (on ?j) (and (between ?j ?j) (and (and (not (and (and (in ?d) (in_motion ?h) ) (in pink) (in_motion ?d ?h) (and (not (same_color ?d) ) ) ) ) (in_motion ?h) ) (adjacent ?j) ) (in_motion ?j) ) ) ) ) ) ) )
                (hold-while (agent_holds ?j ?d) (in_motion ?d) (and (> (distance_side agent ?h) 1) (in_motion ?h ?d) (and (not (agent_holds ?d) ) (not (< (distance ?h ?d) 1) ) ) ) )
                (hold-while (on ?j ?h) (and (in_motion ?j ?d) (in_motion ?h ?h) ) )
              )
            )
          )
        )
      )
    )
    (forall (?f - triangular_ramp)
      (and
        (preference preferenceB
          (exists (?n - hexagonal_bin ?l - hexagonal_bin)
            (at-end
              (agent_holds ?f)
            )
          )
        )
        (preference preferenceC
          (then
            (hold-while (in_motion agent ?f) (adjacent ?f) (and (in_motion ?f) (object_orientation ?f ?f) ) (in ?f ?f) )
            (hold (not (not (in desk ?f) ) ) )
            (hold (in ?f) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 3 180 )
    (< 18 (count-nonoverlapping preferenceA:yellow_pyramid_block) )
  )
)
(:scoring
  (+ (+ 10 (count-nonoverlapping preferenceC:dodgeball) )
    (* 10 (count-nonoverlapping preferenceB:wall:basketball:pink) )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?e - red_dodgeball)
    (exists (?p - shelf)
      (and
        (and
          (exists (?y - dodgeball)
            (exists (?l - block ?h - color)
              (and
                (game-conserved
                  (< (distance ?e) (x_position 7 9))
                )
              )
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - block)
      (and
        (preference preferenceA
          (exists (?e - (either cube_block yellow_cube_block))
            (exists (?l - shelf ?z ?g ?r ?w ?y ?l - (either tall_cylindrical_block dodgeball dodgeball) ?c - curved_wooden_ramp)
              (then
                (once (exists (?q - (either cube_block yellow_cube_block)) (and (exists (?x - (either dodgeball yellow_cube_block)) (and (on ?f ?x) (and (agent_holds ?e) (agent_holds desk left) (not (on ?x) ) (or (not (adjacent ?q) ) (exists (?a - hexagonal_bin) (not (same_color ?q ?a) ) ) ) ) ) ) (in_motion ?c ?c) ) ) )
                (once (<= (distance ?e ?e) (distance bed)) )
                (once (agent_holds ?e) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (count-once-per-objects preferenceA:hexagonal_bin) (count-once preferenceA:basketball:hexagonal_bin:pink) 10 7 (count-once-per-objects preferenceA:red) (count-nonoverlapping preferenceA:beachball) (+ (>= (count-once-per-objects preferenceA:pink) 2 )
      )
      (count-nonoverlapping preferenceA)
      (+ (count-nonoverlapping preferenceA:dodgeball) (- (count-same-positions preferenceA:pink_dodgeball) )
      )
    )
    (* (count-nonoverlapping preferenceA:green:red) 2 )
  )
)
(:scoring
  (* (* (count-nonoverlapping preferenceA) 8 )
    3
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?n - watch ?t - chair ?p - (either blue_cube_block basketball top_drawer))
    (game-conserved
      (in_motion ?p)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once (on ?xxx ?xxx) )
        (once (in_motion ?xxx) )
        (hold (in_motion ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 18 2 )
)
(:scoring
  (count-nonoverlapping preferenceA:green)
)
)

