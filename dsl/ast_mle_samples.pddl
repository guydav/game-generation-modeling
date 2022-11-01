(define (game game-id) (:domain domain-name)
(:setup
  (exists (?z - (either dodgeball yellow))
    (or
      (or
        (game-conserved
          (in ?z pink)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?u - dodgeball)
      (and
        (preference preferenceA
          (then
            (once (touch ?u ?u) )
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
      (exists (?b - doggie_bed)
        (exists (?g - dodgeball)
          (exists (?m - hexagonal_bin)
            (then
              (once (and (<= 1 (distance agent bed)) (in brown) (in ?b) ) )
              (once (on ?b ?m) )
              (once (agent_holds ?m) )
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
  (exists (?z - flat_block)
    (and
      (and
        (and
          (game-conserved
            (not
              (in_motion ?z)
            )
          )
        )
      )
      (forall (?d - shelf)
        (and
          (and
            (forall (?l - cylindrical_block)
              (game-conserved
                (agent_holds ?z)
              )
            )
          )
        )
      )
      (forall (?m - teddy_bear)
        (and
          (and
            (game-conserved
              (not
                (in_motion ?z)
              )
            )
            (forall (?p ?j ?v - ball)
              (exists (?l - sliding_door)
                (and
                  (exists (?h - (either laptop pillow) ?o - bridge_block)
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
    (forall (?m - block)
      (and
        (preference preferenceA
          (exists (?k - chair)
            (then
              (hold (not (in ?k agent) ) )
              (once (in_motion ?k ?k) )
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
                  (exists (?k - game_object ?e - ball)
                    (and
                      (or
                        (in ?e ?e)
                        (agent_holds agent)
                      )
                      (<= (distance ?e bed ?e) 5)
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
      (exists (?r - hexagonal_bin)
        (game-conserved
          (not
            (open ?r ?r)
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
      (exists (?u - pillow)
        (at-end
          (and
            (is_setup_object ?u)
            (in_motion ?u)
            (on upside_down)
          )
        )
      )
    )
    (forall (?d - dodgeball)
      (and
        (preference preferenceB
          (then
            (once (in_motion ?d ?d) )
            (once (and (not (agent_holds agent) ) (not (agent_holds ?d ?d) ) (in agent) ) )
            (hold (and (in ?d) (agent_holds ?d) (agent_holds ?d) (= (distance bed ?d)) ) )
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
    (forall (?i - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?h - (either golfball rug tall_cylindrical_block))
            (exists (?n - hexagonal_bin ?p - hexagonal_bin)
              (then
                (once (in_motion ?i) )
                (once (and (not (not (and (and (adjacent ?h front) (not (not (in_motion ?p) ) ) (agent_holds ?p) ) (touch ?h pink_dodgeball) ) ) ) (in_motion ?i ?i) (agent_holds floor) ) )
                (once (and (on ?h ?h) (and (in ?i) (not (and (in_motion ?i ?i ?p) (on right ?p) (and (and (not (agent_holds ?h) ) (not (adjacent ?p) ) ) (on ?h) (not (agent_holds ?h ?p) ) (in ?h) ) (and (= 6 4 1) (and (equal_z_position ?h ?i) (not (and (in ?h ?p) (touch ?p) (in desk ?h) (game_over ?h ?i) ) ) ) ) (touch ?h) (and (agent_holds ?h) (in_motion ?i) ) (agent_holds ?h) ) ) (and (on ?p ?i) (agent_holds ?p) ) (< 1 (distance 1 ?p)) ) ) )
              )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?e - dodgeball)
        (then
          (once (and (agent_holds ?e rug) (touch ?e) ) )
          (once (agent_holds ?e) )
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
  (exists (?y - red_dodgeball)
    (game-conserved
      (on ?y agent)
    )
  )
)
(:constraints
  (and
    (forall (?a ?m ?b - (either cylindrical_block cube_block key_chain))
      (and
        (preference preferenceA
          (then
            (once (and (and (exists (?n - cube_block ?f - hexagonal_bin) (not (adjacent agent ?b) ) ) (and (adjacent ?a) (touch ?b) ) ) (in_motion agent ?m) ) )
            (once (in_motion ?m) )
            (hold (in ?b) )
          )
        )
        (preference preferenceB
          (exists (?c - teddy_bear)
            (exists (?n - hexagonal_bin)
              (exists (?l - red_dodgeball)
                (exists (?d - shelf ?q - building)
                  (exists (?z - (either blue_cube_block golfball) ?k - (either yellow_cube_block))
                    (exists (?y - teddy_bear)
                      (then
                        (once (agent_holds ?m) )
                        (once (< 0.5 1) )
                        (hold-for 10 (in_motion bed ?l) )
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
      (exists (?p - cube_block)
        (then
          (once (and (agent_holds ?p floor) (not (equal_z_position ?p) ) ) )
          (once (and (and (and (not (in_motion bed ?p) ) (agent_holds top_shelf) ) (and (not (touch ?p front_left_corner) ) (<= (distance agent room_center agent) 2) ) ) (in_motion ?p) ) )
          (hold (and (< (distance ?p 4) 1) (on ?p) (in ?p) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (and (exists (?y - beachball) (agent_holds ?y) ) (in_motion rug ?xxx) (<= 10 (distance_side 9 agent)) (not (in sideways ?xxx) ) ) )
        (hold (and (agent_holds ?xxx) (< (distance 5 ?xxx) 0.5) ) )
        (once (touch ?xxx) )
      )
    )
    (forall (?p - ball)
      (and
        (preference preferenceC
          (exists (?e - (either laptop hexagonal_bin))
            (exists (?f - shelf)
              (then
                (once (exists (?n - (either cellphone blue_cube_block dodgeball) ?h - curved_wooden_ramp) (in_motion ?f) ) )
                (once (< (distance ?p ?f) 0.5) )
              )
            )
          )
        )
        (preference preferenceD
          (exists (?g - pillow)
            (then
              (hold-while (and (not (agent_holds ?p) ) (not (agent_holds ?g agent) ) ) (not (agent_holds ?g ?g) ) )
              (hold (not (not (not (on tan) ) ) ) )
              (once (and (in agent ?p) (on ?g) (not (= (distance desk 9 room_center) 4) ) ) )
            )
          )
        )
        (preference preferenceE
          (exists (?q ?n ?e ?m ?u ?k - (either doggie_bed alarm_clock))
            (exists (?h - curved_wooden_ramp)
              (then
                (once (= (distance ?h ?p) 1 (distance ?e 10)) )
                (once (agent_holds ?n ?q) )
                (once (adjacent ?n) )
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
                      (exists (?d - (either pen doggie_bed doggie_bed dodgeball))
                        (on ?d ?d)
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
    (forall (?g - hexagonal_bin)
      (game-conserved
        (in ?g ?g)
      )
    )
    (forall (?y ?z - hexagonal_bin)
      (exists (?e - hexagonal_bin)
        (and
          (forall (?q - ball)
            (game-optional
              (in_motion ?q)
            )
          )
          (exists (?u - chair)
            (exists (?g - building)
              (exists (?n - (either cd beachball book dodgeball))
                (game-conserved
                  (in ?z desk)
                )
              )
            )
          )
        )
      )
    )
    (forall (?n - (either cylindrical_block dodgeball))
      (and
        (game-conserved
          (on ?n)
        )
        (game-conserved
          (touch ?n ?n)
        )
        (and
          (and
            (game-conserved
              (is_setup_object ?n ?n)
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
      (exists (?t - game_object ?y - hexagonal_bin)
        (exists (?g - (either teddy_bear mug dodgeball))
          (exists (?q - hexagonal_bin ?j - (either dodgeball blue_cube_block pyramid_block yellow))
            (then
              (once (same_color ?j ?j) )
              (once (and (in ?g) (adjacent ?j ?g) (and (and (on ?g ?y) (and (and (and (same_color ?j) (agent_holds ?j) ) (in_motion ?j) ) (and (not (same_color green_golfball ?y) ) (exists (?a - (either triangular_ramp)) (on ?g) ) ) ) ) (and (in_motion ?j) (not (in ?j ?g) ) ) ) (and (or (and (agent_holds bed ?g) (in_motion ?y) ) (not (in_motion ?y ?g) ) ) (agent_holds ?g) ) ) )
              (once (in_motion rug ?g) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?z - hexagonal_bin)
        (exists (?r ?x ?k ?s - shelf ?k - dodgeball ?a - hexagonal_bin)
          (then
            (hold-while (not (on ?a) ) (in_motion ?a) )
            (once (and (agent_holds ?a) (exists (?h - (either mug hexagonal_bin) ?f - curved_wooden_ramp) (and (agent_holds ?a ?f) (not (and (in rug) (not (adjacent_side ?f ?f) ) (and (exists (?y - (either yellow_cube_block golfball)) (not (on desk bed) ) ) (adjacent agent) ) (adjacent ?a ?z) ) ) ) ) (agent_holds ?a ?z) (agent_holds ?z ?z) (and (not (on agent ?z) ) (not (in_motion ?a agent) ) ) (same_type ?a ?z) (not (on ?z) ) ) )
            (once (in_motion ?z ?z) )
          )
        )
      )
    )
    (forall (?s - hexagonal_bin ?r - ball ?h - hexagonal_bin)
      (and
        (preference preferenceC
          (exists (?c - cube_block ?a - hexagonal_bin ?u - (either book pyramid_block) ?t - cylindrical_block)
            (then
              (hold (agent_holds ?h) )
              (forall-sequence (?l - cylindrical_block ?f - curved_wooden_ramp)
                (then
                  (hold-while (in ?h agent) (on ?h ?t) )
                  (once (and (agent_holds ?t) (not (agent_holds agent) ) ) )
                  (once (exists (?k - dodgeball) (on desk) ) )
                )
              )
              (hold-while (and (not (and (on ?t) (and (not (or (not (not (on ?t) ) ) (and (and (agent_holds ?t ?t) (and (or (= 0.5 1) (>= 1 4) ) (in ?h) (and (and (agent_holds ?h agent) (agent_holds ?h ?t) ) (not (in ?h) ) ) ) ) (same_color ?h) ) ) ) (not (on ?t) ) ) ) ) (in_motion ?h ?h) (and (in rug) (in_motion ?h) ) (in ?h ?h) (in_motion ?t) (and (and (not (> 1 2) ) (and (on ?t) (not (and (in_motion ?t) (agent_holds ?h) ) ) ) ) (agent_crouches ?t ?t) ) ) (and (agent_holds ?h) (on ?h) ) )
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
      (forall (?y - dodgeball ?a - game_object)
        (and
          (forall (?u - beachball ?g - hexagonal_bin)
            (forall (?w - block)
              (and
                (forall (?q - (either yellow pillow))
                  (exists (?y - hexagonal_bin)
                    (and
                      (game-optional
                        (and
                          (touch ?w)
                          (not
                            (and
                              (= (distance ?a ?w 10) (distance ))
                              (not
                                (not
                                  (agent_holds ?g)
                                )
                              )
                              (not
                                (adjacent_side ?g ?q)
                              )
                            )
                          )
                        )
                      )
                      (game-optional
                        (not
                          (and
                            (in_motion ?g)
                            (on ?a)
                          )
                        )
                      )
                      (exists (?c - pyramid_block)
                        (or
                          (and
                            (exists (?x - book ?v - dodgeball)
                              (game-conserved
                                (not
                                  (and
                                    (on ?q)
                                    (and
                                      (in_motion ?y ?v ?a)
                                      (and
                                        (< 1 4)
                                        (and
                                          (not
                                            (and
                                              (in ?c)
                                              (in ?c)
                                              (not
                                                (and
                                                  (agent_holds ?v)
                                                  (and
                                                    (and
                                                      (game_over ?a)
                                                      (not
                                                        (in_motion ?v ?c)
                                                      )
                                                    )
                                                    (= 1 (distance 7 7))
                                                  )
                                                  (adjacent ?w)
                                                )
                                              )
                                            )
                                          )
                                          (not
                                            (in_motion ?y rug)
                                          )
                                        )
                                      )
                                    )
                                    (and
                                      (in_motion ?y)
                                      (not
                                        (< 1 (distance agent ?g))
                                      )
                                    )
                                  )
                                )
                              )
                            )
                          )
                          (or
                            (game-conserved
                              (in_motion ?q)
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
            (exists (?q - red_dodgeball)
              (forall (?s - hexagonal_bin)
                (exists (?k - block)
                  (and
                    (exists (?g ?d - wall)
                      (and
                        (game-conserved
                          (and
                            (agent_holds ?a)
                            (= (distance_side ?g agent))
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
            (and
              (exists (?x - dodgeball)
                (and
                  (forall (?g - hexagonal_bin)
                    (game-conserved
                      (in_motion ?g ?x)
                    )
                  )
                )
              )
              (game-conserved
                (and
                  (agent_holds ?a agent)
                  (agent_holds ?a front)
                )
              )
            )
            (game-conserved
              (and
                (or
                  (and
                    (and
                      (agent_holds ?a)
                    )
                    (not
                      (touch ?a)
                    )
                  )
                  (forall (?s - red_dodgeball)
                    (adjacent ?s ?a)
                  )
                )
                (and
                  (in ?a ?a)
                  (not
                    (agent_holds ?a ?a)
                  )
                  (is_setup_object ?a ?a)
                )
                (agent_holds ?a ?a)
                (object_orientation ?a)
              )
            )
          )
          (exists (?j ?y - (either dodgeball hexagonal_bin rug))
            (exists (?o - dodgeball)
              (and
                (exists (?v - dodgeball ?m - hexagonal_bin)
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
  (exists (?s - (either golfball lamp wall dodgeball key_chain cd cylindrical_block))
    (exists (?o ?m - (either laptop golfball))
      (game-conserved
        (or
          (not
            (and
              (not
                (< 10 6)
              )
              (exists (?h ?q - game_object)
                (agent_holds ?h ?q)
              )
            )
          )
          (agent_holds ?o)
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
      (exists (?b - hexagonal_bin)
        (game-optional
          (and
            (touch ?b)
            (not
              (agent_holds ?b)
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
          (forall (?d - dodgeball)
            (forall (?l - (either doggie_bed doggie_bed) ?w - hexagonal_bin)
              (forall (?n - wall)
                (exists (?z - ball)
                  (exists (?j - hexagonal_bin ?x - hexagonal_bin ?u - chair)
                    (game-optional
                      (in ?z ?n)
                    )
                  )
                )
              )
            )
          )
          (exists (?a ?s - block ?n - hexagonal_bin)
            (and
              (exists (?y - cube_block)
                (game-optional
                  (touch floor)
                )
              )
            )
          )
          (forall (?e - (either blue_cube_block pyramid_block) ?c - hexagonal_bin)
            (not
              (forall (?d - (either tall_cylindrical_block laptop) ?w - flat_block)
                (and
                  (and
                    (exists (?x - shelf)
                      (game-conserved
                        (in ?x rug floor)
                      )
                    )
                    (game-conserved
                      (in_motion ?c)
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
        (exists (?w ?g - cube_block)
          (exists (?v - drawer)
            (and
              (game-conserved
                (in ?g ?w)
              )
              (and
                (and
                  (forall (?a - hexagonal_bin)
                    (and
                      (forall (?m - (either bed cd) ?u - ball ?o ?l - pyramid_block)
                        (forall (?e - (either cube_block dodgeball hexagonal_bin watch))
                          (exists (?t - hexagonal_bin)
                            (game-optional
                              (adjacent ?t ?t)
                            )
                          )
                        )
                      )
                    )
                  )
                  (exists (?d - hexagonal_bin)
                    (and
                      (and
                        (exists (?e - triangular_ramp ?y ?k - wall ?e - hexagonal_bin)
                          (exists (?z - hexagonal_bin)
                            (game-conserved
                              (in_motion ?z ?w)
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
              (exists (?b - game_object)
                (and
                  (and
                    (game-optional
                      (and
                        (not
                          (and
                            (in_motion bed)
                            (agent_holds ?v)
                          )
                        )
                        (and
                          (not
                            (not
                              (in ?g bed)
                            )
                          )
                          (in_motion bed ?g)
                        )
                      )
                    )
                    (and
                      (game-optional
                        (and
                          (on ?w)
                          (not
                            (not
                              (not
                                (agent_holds ?v)
                              )
                            )
                          )
                        )
                      )
                      (exists (?m - chair)
                        (game-optional
                          (not
                            (adjacent ?b)
                          )
                        )
                      )
                    )
                  )
                  (not
                    (game-optional
                      (not
                        (touch ?g ?v)
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
      (exists (?g ?d - bridge_block)
        (then
          (once (in_motion agent ?g) )
          (once (and (not (not (in ?g) ) ) (agent_holds ?d ?g) ) )
          (once (forall (?a - (either pyramid_block red)) (in_motion ?a ?d) ) )
        )
      )
    )
    (forall (?c - cube_block ?s - color ?i - red_pyramid_block)
      (and
        (preference preferenceB
          (exists (?t - dodgeball ?w - (either cd dodgeball golfball laptop) ?x - game_object ?b - doggie_bed)
            (then
              (once (in agent ?b) )
              (once (and (in ?i) (in_motion ?i) (exists (?f - cube_block) (on desk) ) ) )
              (once (in_motion ?b ?b) )
            )
          )
        )
        (preference preferenceC
          (then
            (hold (or (and (rug_color_under ?i ?i) (in ?i) (and (and (agent_holds ?i) ) (in ?i) ) (and (not (in_motion ?i) ) (in_motion ?i ?i) ) ) (object_orientation ?i ?i) (same_color ?i front) ) )
            (once (in_motion ?i ?i) )
            (once (not (and (agent_holds upright) (on ?i ?i) ) ) )
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
    (exists (?q - building ?y - (either pyramid_block book))
      (game-optional
        (not
          (agent_holds ?y)
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
    (forall (?r - ball ?l - tall_cylindrical_block)
      (and
        (preference preferenceA
          (then
            (once (not (agent_holds ?l) ) )
            (hold (touch ?l ?l) )
            (once (in ?l ?l) )
            (once (and (exists (?t ?u - hexagonal_bin) (in_motion bed desk) ) (open ?l) ) )
          )
        )
      )
    )
    (forall (?a - dodgeball)
      (and
        (preference preferenceB
          (then
            (hold (exists (?q - red_pyramid_block) (not (in_motion ?a) ) ) )
            (once-measure (on ?a ?a) (distance ?a agent) )
            (once (not (not (and (in_motion ?a brown) (in ?a) ) ) ) )
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
    (exists (?h - hexagonal_bin)
      (on ?h ?h)
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
  (exists (?s - block ?h - building)
    (game-optional
      (and
        (not
          (< 1 (distance 5 room_center))
        )
        (in_motion agent ?h)
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - dodgeball)
      (and
        (preference preferenceA
          (exists (?k ?h ?m ?w ?z ?d - chair)
            (at-end
              (agent_holds ?k)
            )
          )
        )
        (preference preferenceB
          (then
            (once (agent_holds ?f agent) )
            (once (in_motion ?f ?f) )
            (once (in_motion ?f) )
          )
        )
        (preference preferenceC
          (exists (?g - block)
            (at-end
              (adjacent_side ?f ?f)
            )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?b - (either dodgeball dodgeball))
        (then
          (once (agent_holds ?b ?b) )
          (once (agent_holds ?b) )
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
    (forall (?o - hexagonal_bin)
      (and
        (preference preferenceF
          (exists (?k - cube_block)
            (exists (?t ?r ?n - cube_block ?h ?e - (either cylindrical_block hexagonal_bin laptop))
              (then
                (once (in_motion ?e) )
                (once (in_motion ?h) )
                (once (on ?e ?k) )
              )
            )
          )
        )
      )
    )
    (preference preferenceG
      (exists (?p - hexagonal_bin)
        (then
          (once (not (not (not (on agent) ) ) ) )
          (once (or (not (and (in_motion ?p) (in_motion ?p ?p) (exists (?g - curved_wooden_ramp) (or (adjacent ?g) (in_motion bed ?p) ) ) ) ) (exists (?x - game_object) (in_motion ?p ?x ?p) ) ) )
          (once (forall (?t - chair) (on ?p) ) )
          (once (in ?p ?p) )
        )
      )
    )
    (forall (?f - dodgeball ?s - beachball)
      (and
        (preference preferenceH
          (then
            (hold (or (same_object agent ?s) (in ?s) ) )
            (hold (agent_holds desk) )
            (hold (= (building_size ) 0.5 (distance ?s agent)) )
          )
        )
        (preference preferenceI
          (exists (?n - ball)
            (then
              (once (in_motion ?s) )
              (hold (in_motion pink ?n) )
              (once (and (agent_holds ?n ?s) (not (and (> 1 3) ) ) ) )
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
      (exists (?o - hexagonal_bin)
        (then
          (hold (not (and (and (and (forall (?i - hexagonal_bin) (and (not (exists (?p - hexagonal_bin) (agent_holds ?i ?i) ) ) (in ?o ?i) (exists (?f - doggie_bed ?a - game_object) (not (object_orientation agent) ) ) (touch ?i) (agent_holds ?o ?o) (and (in ?i) (in upright floor ?i ?i) ) ) ) (rug_color_under ?o) ) (not (equal_z_position agent) ) ) (agent_holds ?o ?o) ) ) )
          (once (= 2 1) )
          (hold (agent_holds ?o ?o) )
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
    (forall (?m - block)
      (and
        (preference preferenceA
          (then
            (once (agent_holds rug) )
            (once (not (agent_holds ?m) ) )
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
    (exists (?n - block)
      (exists (?x - dodgeball ?f - hexagonal_bin ?d - ball)
        (game-optional
          (not
            (type ?n)
          )
        )
      )
    )
    (exists (?i - (either teddy_bear pen))
      (game-conserved
        (and
          (not
            (touch ?i ?i)
          )
          (same_color ?i)
        )
      )
    )
    (game-conserved
      (exists (?g - hexagonal_bin)
        (not
          (agent_holds ?g ?g)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?w - color)
        (exists (?b - block ?d - dodgeball)
          (exists (?e - dodgeball)
            (exists (?j - cube_block)
              (then
                (once (and (in_motion ?d) (not (adjacent ?j) ) (rug_color_under ?w) ) )
                (once (agent_holds rug ?w) )
                (once (in_motion ?w) )
              )
            )
          )
        )
      )
    )
    (forall (?n ?j - (either key_chain golfball tall_cylindrical_block teddy_bear curved_wooden_ramp cube_block main_light_switch))
      (and
        (preference preferenceB
          (exists (?l - dodgeball)
            (then
              (hold (in_motion ?j agent) )
              (once (agent_holds rug) )
              (hold-while (on ?l ?n) (in_motion ?l) )
            )
          )
        )
      )
    )
    (forall (?c - red_dodgeball ?g - wall ?m - hexagonal_bin)
      (and
        (preference preferenceC
          (exists (?v - game_object)
            (then
              (once (and (agent_holds door floor) (in_motion ?m ?v) ) )
              (once (in_motion blue) )
              (once (same_color ?v ?m pillow ?v) )
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
  (exists (?a - cube_block)
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
  (exists (?s - color)
    (game-conserved
      (in_motion ?s top_shelf)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?d - shelf)
        (exists (?a - (either dodgeball triangle_block pillow) ?o - triangular_ramp)
          (exists (?q - building)
            (then
              (once (adjacent ?d) )
              (once (not (and (not (not (adjacent ?d) ) ) (adjacent pillow) ) ) )
              (hold (not (and (= (distance ?d) (distance ?o ?d)) (agent_holds ?d) ) ) )
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
        (once (exists (?d - (either pyramid_block alarm_clock)) (rug_color_under ?d) ) )
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
      (exists (?f - hexagonal_bin)
        (then
          (once (not (on ?f) ) )
          (once (in_motion desktop ?f ?f) )
          (once (agent_holds ?f ?f) )
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
    (forall (?a - golfball)
      (and
        (preference preferenceC
          (exists (?d - cube_block)
            (exists (?w - golfball ?e - dodgeball)
              (exists (?w - (either pyramid_block pencil))
                (then
                  (once (or (in bed) (in_motion ?e) ) )
                  (hold (same_color ?e) )
                  (once (and (not (not (adjacent rug) ) ) (and (and (in ?a) (not (and (exists (?x - dodgeball) (in_motion ?d ?x) ) (in_motion ?d) ) ) (in_motion ?d west_wall) ) (adjacent ?d ?e) (on ?a) ) (not (in_motion ?e ?e) ) (not (touch ?d ?d) ) (in ?a ?a) (not (in_motion desk) ) (agent_holds door ?e) (agent_holds ?e) ) )
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
  (forall (?f - game_object)
    (forall (?h - (either alarm_clock cylindrical_block))
      (game-optional
        (and
          (in_motion ?h)
          (agent_holds ?f)
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
    (forall (?k - hexagonal_bin ?a - ball)
      (and
        (preference preferenceB
          (exists (?r - dodgeball ?x - hexagonal_bin)
            (then
              (once (not (not (and (agent_holds pink desk) (in ?a ?x) ) ) ) )
              (once (and (in_motion ?x ?a) (agent_holds ?x) ) )
              (once (not (type ?x) ) )
              (once (not (in_motion agent ?a) ) )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?v - pillow)
        (then
          (once (adjacent_side ?v) )
          (once (object_orientation ?v) )
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
  (exists (?x - ball)
    (forall (?m ?y - hexagonal_bin)
      (game-conserved
        (forall (?q - game_object)
          (in_motion ?x ?x)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?k - ball)
        (exists (?w - ball)
          (then
            (hold (in_motion ?w) )
            (once (in_motion ?k) )
            (hold (in_motion ?k) )
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
      (exists (?q - doggie_bed)
        (at-end
          (and
            (object_orientation back ?q)
            (on ?q)
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (not (exists (?t - cube_block) (< (distance ) 3) ) ) )
        (once (in_motion ?xxx ?xxx) )
        (once (in ?xxx) )
      )
    )
    (preference preferenceC
      (exists (?v - building)
        (at-end
          (not
            (in_motion ?v)
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
    (forall (?q - block)
      (and
        (preference preferenceA
          (then
            (once (< 6 1) )
            (once (and (not (agent_holds ?q ?q) ) (agent_holds ?q ?q) ) )
            (once (exists (?g - building) (on ?g ?q) ) )
          )
        )
        (preference preferenceB
          (then
            (once-measure (in_motion ?q ?q) (distance 2 door) )
            (once (not (type rug) ) )
            (once (same_color ?q ?q) )
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
  (forall (?g ?i ?x - teddy_bear ?v - wall)
    (exists (?t - (either pyramid_block alarm_clock))
      (exists (?g - dodgeball ?h - hexagonal_bin)
        (game-optional
          (and
            (and
              (same_color ?v)
              (< 1 (distance ))
              (exists (?r - game_object)
                (< 1 1)
              )
              (in_motion ?t agent)
            )
            (on ?h right)
            (not
              (agent_holds ?t)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?j - block ?e - drawer ?y - book ?v - hexagonal_bin ?q - ball)
      (and
        (preference preferenceA
          (exists (?d - drawer ?x ?n ?b - triangular_ramp)
            (exists (?e - dodgeball)
              (then
                (hold (and (agent_holds rug) (agent_holds ?e) ) )
                (once (< 2 4) )
                (once (on ?x ?e) )
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
      (exists (?v - ball)
        (then
          (hold-while (and (in ?v) (and (and (and (< (distance ?v) (distance ?v ?v)) (on ?v ?v) ) (in ?v ?v) ) (and (in_motion left ?v) (not (between ?v) ) ) (in agent) ) (and (on ?v) (and (adjacent_side ?v ?v) (not (and (and (in desk) (on ?v) (agent_holds ?v) ) (on pillow) (not (on ?v ?v) ) (and (and (exists (?b - dodgeball) (adjacent ?b ?v) ) (agent_holds ?v ?v) ) (and (> 2 (distance room_center door)) (in ?v) ) (and (and (rug_color_under ?v) (adjacent ?v) ) (> (distance ?v 9) 2) ) ) (in_motion pink) (in ?v) (and (adjacent ?v) (and (touch ?v ?v) ) ) ) ) ) ) (faces ?v) ) (not (same_color ?v) ) )
          (once (equal_z_position ?v ?v agent) )
          (once (not (in ?v ?v) ) )
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
    (forall (?d - dodgeball)
      (and
        (preference preferenceA
          (then
            (once (rug_color_under ?d ?d) )
            (once (agent_holds ?d ?d ?d) )
            (once (in_motion ?d) )
          )
        )
        (preference preferenceB
          (then
            (once (and (touch ?d) (agent_holds pink) ) )
            (hold-while (forall (?s - (either cellphone desktop mug dodgeball yellow desktop dodgeball)) (adjacent ?s) ) (in_motion ?d) (on ?d) )
            (once (and (in ?d) (and (on desk floor) (in ?d) ) ) )
          )
        )
      )
    )
    (forall (?i - golfball)
      (and
        (preference preferenceC
          (exists (?c - game_object)
            (exists (?e - doggie_bed)
              (exists (?y - drawer)
                (then
                  (hold (in_motion agent ?i) )
                  (once (and (agent_holds ?i) (on ?y) ) )
                  (hold (agent_holds ?y ?y) )
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
      (exists (?r - dodgeball)
        (then
          (once (agent_holds ?r ?r) )
          (hold (in_motion ?r ?r) )
          (hold-while (agent_holds ?r ?r) (agent_holds ?r) )
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
  (exists (?y - drawer)
    (game-conserved
      (and
        (in_motion ?y ?y)
        (in_motion green_golfball)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?m ?i - (either cd golfball blue_cube_block yellow_cube_block))
        (exists (?z - hexagonal_bin)
          (then
            (once (not (and (in desk) (and (above ?i) (not (touch ?z ?i) ) ) ) ) )
            (hold-while (agent_holds ?z) (or (not (in_motion ?i ?m) ) ) (and (exists (?u - block) (not (and (agent_holds ?z ?m) (in_motion ?u ?z) ) ) ) (same_type ?m) (in ?m) ) )
            (once (not (not (and (in ?i yellow) (agent_holds ?i) ) ) ) )
          )
        )
      )
    )
    (forall (?r - teddy_bear ?x - game_object)
      (and
        (preference preferenceB
          (then
            (once (not (adjacent ) ) )
            (hold (in_motion ?x ?x) )
            (hold (not (in_motion floor ?x) ) )
          )
        )
        (preference preferenceC
          (at-end
            (on ?x ?x)
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
    (forall (?a - dodgeball)
      (forall (?h - hexagonal_bin)
        (forall (?l - tall_cylindrical_block)
          (exists (?v - hexagonal_bin)
            (game-conserved
              (and
                (adjacent ?l ?v)
                (on ?v ?v)
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
    (forall (?s - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?d - (either blue_cube_block wall))
            (exists (?q - shelf)
              (at-end
                (not
                  (in_motion ?s)
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
  (exists (?j - (either tall_cylindrical_block curved_wooden_ramp) ?d - shelf ?a - teddy_bear)
    (forall (?c - curved_wooden_ramp)
      (game-conserved
        (agent_holds ?a)
      )
    )
  )
)
(:constraints
  (and
    (forall (?m - cube_block)
      (and
        (preference preferenceA
          (exists (?l - blue_pyramid_block ?f - tall_cylindrical_block ?i - building ?b - teddy_bear)
            (exists (?x - dodgeball)
              (then
                (hold-while (and (in_motion ?m ?b) (exists (?u - hexagonal_bin ?e - dodgeball) (and (agent_holds ?e ?x) (and (not (in_motion desktop) ) (not (not (and (in_motion ?e) (touch ?e) ) ) ) ) ) ) ) (and (not (on ?b) ) (not (< 5 1) ) (agent_holds ?m ?b) ) )
                (once (not (agent_holds ?x rug) ) )
                (hold (and (< (distance ?b ?m) (distance ?x desk)) (not (and (adjacent_side side_table ?b) (exists (?f - hexagonal_bin) (adjacent ?f) ) ) ) ) )
              )
            )
          )
        )
      )
    )
    (forall (?z - hexagonal_bin)
      (and
        (preference preferenceB
          (exists (?w - rug)
            (exists (?b - hexagonal_bin ?t - (either tall_cylindrical_block triangle_block))
              (exists (?q ?n - hexagonal_bin)
                (at-end
                  (in_motion ?w)
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
              (exists (?k - hexagonal_bin)
                (and
                  (and
                    (not
                      (adjacent ?k)
                    )
                    (not
                      (in_motion ?k ?k)
                    )
                    (>= 9 1)
                    (on ?k)
                  )
                  (and
                    (in_motion ?k ?k)
                    (in ?k ?k)
                    (not
                      (between ?k ?k yellow)
                    )
                  )
                )
              )
            )
          )
        )
        (exists (?r ?m - book)
          (game-optional
            (exists (?g - cube_block ?y - red_pyramid_block)
              (and
                (not
                  (in_motion left ?m)
                )
                (agent_holds ?y)
                (and
                  (agent_holds ?r ?m)
                  (or
                    (agent_holds bed)
                    (and
                      (in ?r)
                      (on ?y)
                    )
                    (agent_holds ?r)
                  )
                )
              )
            )
          )
        )
        (forall (?f - (either pencil))
          (exists (?n - hexagonal_bin ?m - cube_block)
            (game-optional
              (in_motion ?m agent)
            )
          )
        )
      )
      (forall (?a - doggie_bed)
        (or
          (game-conserved
            (agent_holds ?a)
          )
          (game-optional
            (adjacent agent ?a)
          )
        )
      )
      (and
        (and
          (exists (?a - (either key_chain beachball yellow_cube_block))
            (and
              (and
                (and
                  (forall (?g - hexagonal_bin)
                    (and
                      (game-optional
                        (and
                          (agent_holds bed ?g)
                          (and
                            (agent_holds ?a)
                            (not
                              (not
                                (not
                                  (and
                                    (forall (?y - ball)
                                      (agent_holds rug)
                                    )
                                    (not
                                      (agent_holds agent)
                                    )
                                  )
                                )
                              )
                            )
                            (in ?a)
                            (on ?a)
                            (on ?a ?g ?g)
                            (in_motion desk)
                            (on ?g blue)
                            (exists (?w - curved_wooden_ramp)
                              (and
                                (and
                                  (agent_holds ?g)
                                  (on ?a ?w)
                                )
                              )
                            )
                            (and
                              (exists (?l - hexagonal_bin)
                                (not
                                  (in_motion ?l ?g)
                                )
                              )
                              (agent_holds pink_dodgeball)
                              (> (distance room_center ?a) (distance room_center 2))
                            )
                            (and
                              (in bed)
                              (in upside_down ?g floor)
                            )
                            (in desk)
                            (agent_holds ?a ?g)
                          )
                          (in ?g)
                        )
                      )
                      (game-conserved
                        (= (distance ?g ?a) 1)
                      )
                      (exists (?n - game_object)
                        (and
                          (exists (?h - golfball)
                            (game-optional
                              (agent_holds ?g)
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (and
                  (game-conserved
                    (agent_holds ?a ?a)
                  )
                )
              )
              (forall (?q - hexagonal_bin)
                (and
                  (forall (?o - hexagonal_bin)
                    (and
                      (exists (?h - (either beachball desktop))
                        (and
                          (forall (?t - hexagonal_bin)
                            (and
                              (game-optional
                                (and
                                  (and
                                    (and
                                      (agent_holds ?q)
                                      (and
                                        (on floor)
                                        (on ?h ?o)
                                      )
                                    )
                                    (agent_holds ?o ?h)
                                  )
                                  (agent_holds ?a)
                                )
                              )
                              (exists (?z - hexagonal_bin)
                                (game-conserved
                                  (and
                                    (and
                                      (in_motion ?h ?o)
                                      (in ?o top_shelf)
                                    )
                                    (adjacent ?o)
                                  )
                                )
                              )
                              (game-optional
                                (in_motion ?q ?t)
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
                    (forall (?p - doggie_bed)
                      (exists (?u - hexagonal_bin ?q - chair)
                        (game-conserved
                          (in_motion ?p)
                        )
                      )
                    )
                  )
                )
                (forall (?y - (either chair))
                  (and
                    (game-optional
                      (in_motion ?y ?y ?a)
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
      (exists (?b - wall ?j - hexagonal_bin ?j - hexagonal_bin ?v - dodgeball)
        (exists (?d - dodgeball)
          (exists (?m - hexagonal_bin)
            (then
              (once (in_motion agent ?d) )
              (hold (in_motion pink) )
              (once (exists (?t - (either credit_card key_chain)) (and (not (not (on ?t) ) ) (and (same_color ?v agent) (same_color ?d) ) ) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?e - hexagonal_bin)
        (exists (?j - hexagonal_bin ?m ?p - game_object)
          (then
            (once (and (exists (?u - hexagonal_bin) (agent_holds floor sideways) ) (not (and (< (distance ?p 8) 1) (and (in_motion ?p ?e ?e) (on ?e) (type ?m ?p) (agent_holds agent ?p) ) (on ?e) (is_setup_object agent) ) ) ) )
            (once (not (or (and (and (agent_holds ?m rug) (agent_holds ?m ?p) ) (or (agent_holds ?m) (< (distance 6) (distance ?p ?e)) ) (not (in ?e ?m) ) ) (or (< 0.5 0.5) (in ?p) ) ) ) )
            (once (on ?p) )
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
      (exists (?v - dodgeball)
        (then
          (once (not (and (not (in_motion ?v) ) (not (agent_holds ?v) ) ) ) )
          (hold (in_motion ?v) )
          (once (and (is_setup_object ?v ?v) (exists (?u - block) (in_motion ?v ?u) ) (and (agent_holds ?v) (agent_holds ?v) ) ) )
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
      (exists (?w ?g - building ?x - doggie_bed)
        (exists (?t - hexagonal_bin ?j - (either hexagonal_bin cellphone))
          (exists (?d - building)
            (then
              (hold-while (in_motion ?d ?j) (on rug ?j) )
              (once (not (in_motion ?j) ) )
              (hold (agent_holds ?d ?d) )
            )
          )
        )
      )
    )
    (preference preferenceE
      (then
        (once (agent_holds ?xxx) )
        (hold-while (and (exists (?v - chair) (on ?v) ) (faces ?xxx ?xxx) ) (in bed ?xxx) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
    (preference preferenceF
      (exists (?j - (either cd alarm_clock) ?s - hexagonal_bin ?k - curved_wooden_ramp ?y - hexagonal_bin)
        (exists (?i - hexagonal_bin ?l - pyramid_block)
          (then
            (once (in_motion ?l) )
            (once (and (is_setup_object ?l front) (and (in_motion ?y ?y) (not (and (agent_holds ?l ?y) (in desk) (< 1 (x_position ?l 3)) (or (in_motion ?y) (and (in_motion ?l ?l ?l) (and (exists (?b - dodgeball ?n ?r ?j - dodgeball ?s ?o - curved_wooden_ramp ?a - building) (<= 2 5) ) (< 2 (distance ?y agent)) ) ) ) (on ?y) (not (not (in ?l) ) ) (in_motion ?y) (exists (?h - (either hexagonal_bin bridge_block)) (on ?l) ) ) ) ) ) )
            (hold (in_motion ?l) )
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
  (exists (?q - teddy_bear ?i - chair)
    (or
      (game-optional
        (or
          (in_motion ?i ?i)
          (in ?i ?i)
          (agent_holds ?i ?i)
        )
      )
      (exists (?j - hexagonal_bin)
        (and
          (exists (?m - hexagonal_bin ?k - hexagonal_bin)
            (game-optional
              (type ?j ?j)
            )
          )
        )
      )
      (and
        (game-conserved
          (in ?i)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?a - block)
        (then
          (once (and (not (in rug) ) (on ?a) (in_motion ?a) ) )
          (hold (in ?a) )
          (hold (and (and (exists (?v - cube_block ?h - cube_block) (touch ?h ?a) ) (agent_holds ?a) ) (agent_holds ?a) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (in_motion ?xxx) )
        (hold-while (and (exists (?g - teddy_bear ?g ?b - bridge_block) (and (agent_holds ?b ?b) (in agent) (equal_x_position ?g) (and (on ?g) (object_orientation bridge_block ?g) ) (not (in_motion ?b) ) (and (and (on ?b) (agent_holds ?g) ) (same_color ?b) (agent_holds ?g) ) (not (same_color ?g ?b) ) (agent_holds agent green) (on ?g) (and (on agent ?b) (in_motion agent ?b) ) ) ) (on ?xxx ?xxx desk ?xxx) ) (and (agent_holds desk) (= (distance 7 ?xxx) (distance bed ?xxx)) ) (in_motion ?xxx) )
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
                (exists (?x - hexagonal_bin)
                  (exists (?q - curved_wooden_ramp)
                    (in_motion rug ?q)
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
  (exists (?e - dodgeball)
    (and
      (game-conserved
        (agent_holds ?e)
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - teddy_bear)
      (and
        (preference preferenceA
          (at-end
            (and
              (not
                (not
                  (on ?r)
                )
              )
              (agent_holds ?r ?r)
              (in_motion ?r)
            )
          )
        )
        (preference preferenceB
          (exists (?f - blinds ?v ?l - yellow_cube_block)
            (then
              (once (and (and (agent_holds ?v ?v) (forall (?a ?b - hexagonal_bin) (in_motion ?a ?b) ) (touch ?r top_shelf) ) (between ?v) ) )
              (once (in_motion ?v) )
              (any)
            )
          )
        )
        (preference preferenceC
          (exists (?a - golfball ?m - cube_block)
            (exists (?w ?a - doggie_bed)
              (then
                (once (in desk ?w) )
                (hold (on ?a) )
                (once (in_motion ?w ?w) )
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
  (forall (?l - (either chair cd cellphone))
    (game-optional
      (and
        (agent_holds ?l)
        (in ?l ?l)
        (in_motion ?l)
        (type ?l)
        (in_motion ?l)
        (exists (?b - dodgeball)
          (adjacent ?l ?b ?b)
        )
        (not
          (and
            (agent_holds ?l)
            (not
              (not
                (forall (?c - teddy_bear)
                  (not
                    (and
                      (not
                        (and
                          (in_motion right)
                          (and
                            (not
                              (agent_holds ?l)
                            )
                            (in ?l)
                          )
                          (not
                            (and
                              (not
                                (and
                                  (in_motion ?l)
                                  (not
                                    (in ?c ?c)
                                  )
                                )
                              )
                              (touch ?l agent)
                              (agent_holds pink_dodgeball ?c)
                            )
                          )
                        )
                      )
                      (in_motion ?c)
                    )
                  )
                )
              )
            )
          )
        )
        (in ?l)
        (in_motion bed ?l)
        (or
          (agent_holds ?l sideways)
        )
        (not
          (agent_holds rug)
        )
        (agent_holds ?l)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?p - hexagonal_bin)
        (exists (?r - ball ?c - curved_wooden_ramp)
          (exists (?m - dodgeball ?a - building)
            (exists (?b - teddy_bear)
              (exists (?j - dodgeball)
                (at-end
                  (and
                    (and
                      (adjacent bed)
                      (exists (?h - ball)
                        (in_motion ?p)
                      )
                    )
                    (and
                      (< (distance room_center ?j) 1)
                      (and
                        (and
                          (agent_holds agent ?p)
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
      (exists (?z - cylindrical_block)
        (then
          (once (< 1 (distance 7 ?z)) )
          (once (agent_holds south_wall ?z) )
          (once (not (game_start ?z) ) )
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
        (once (exists (?n - cube_block ?c - cube_block ?s - ball) (in_motion ?s ?s) ) )
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
  (forall (?d - (either lamp dodgeball key_chain) ?f - (either book red))
    (game-conserved
      (in_motion ?f desk ?f)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?a ?i - ball ?k - (either dodgeball cube_block) ?s - red_pyramid_block)
        (at-end
          (agent_holds ?s)
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
  (exists (?p ?q ?x - pyramid_block ?u - hexagonal_bin)
    (game-conserved
      (exists (?o - block ?g - doggie_bed ?b - hexagonal_bin)
        (in_motion front ?b)
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - cube_block)
      (and
        (preference preferenceA
          (then
            (once (and (not (same_object ?a ?a) ) (on ?a ?a) ) )
            (once (agent_holds ?a) )
            (hold (not (adjacent agent ?a) ) )
          )
        )
        (preference preferenceB
          (then
            (hold (on ?a) )
            (hold (in_motion ?a) )
            (once (not (not (in_motion ?a) ) ) )
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
    (exists (?p - game_object)
      (and
        (game-optional
          (in_motion ?p ?p)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?e - doggie_bed)
        (at-end
          (on ?e)
        )
      )
    )
    (preference preferenceB
      (exists (?z - dodgeball ?n - hexagonal_bin)
        (then
          (hold-while (not (and (not (and (not (agent_holds ?n ?n) ) (in ?n) ) ) (touch ?n) ) ) (agent_holds ?n ?n) )
          (once (not (agent_holds ?n) ) )
          (hold (and (in_motion agent) (not (and (and (agent_holds ?n) (in_motion ?n) ) (above ) ) ) ) )
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
            (exists (?p - cube_block ?n - dodgeball)
              (exists (?x - dodgeball)
                (and
                  (and
                    (not
                      (and
                        (in_motion ?x)
                        (not
                          (in_motion ?n)
                        )
                        (agent_holds ?x ?n)
                        (game_start ?n rug)
                      )
                    )
                    (>= (distance room_center ?x) (distance ?x ?x))
                    (not
                      (in_motion ?x rug)
                    )
                  )
                  (in ?n)
                  (< 1 (distance room_center agent))
                  (agent_holds top_drawer ?n)
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
      (exists (?b - game_object)
        (then
          (once (and (not (in_motion ?b) ) (forall (?t - cube_block) (in_motion ?t ?t) ) ) )
          (once (same_color ?b ?b) )
          (once (not (in bed ?b) ) )
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
        (once (exists (?q - curved_wooden_ramp ?q - beachball) (not (and (on ?q) (rug_color_under desk) (in floor) ) ) ) )
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
              (forall (?y - hexagonal_bin ?r - (either cylindrical_block pencil))
                (agent_holds ?r ?r)
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
      (exists (?e ?z - game_object)
        (then
          (once (and (and (not (is_setup_object ?e ?e) ) (same_color ?e) (touch brown) ) (not (agent_holds ?e) ) ) )
          (hold (agent_holds agent) )
          (once (not (not (in_motion ?e ?z) ) ) )
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
        (once (and (< (distance 5 room_center) 1) (not (touch ?xxx) ) (not (open upside_down ?xxx ?xxx) ) (and (in_motion ?xxx agent) (not (on ?xxx ?xxx) ) ) (in ?xxx bed) (exists (?r - teddy_bear ?j - building ?m - ball) (in_motion ?m ?m) ) (in_motion bed agent) ) )
      )
    )
    (preference preferenceB
      (then
        (hold (on ?xxx) )
        (hold (and (in_motion ?xxx) (agent_holds ?xxx) (and (and (not (agent_holds ?xxx) ) (not (in ?xxx ?xxx) ) ) (and (not (agent_holds ?xxx ?xxx) ) (exists (?d - ball ?s - red_pyramid_block ?q ?j - (either mug yellow_cube_block)) (same_color ?q) ) (on ?xxx agent) ) ) (and (touch floor) (agent_holds agent) ) (agent_holds ?xxx) (agent_holds agent) ) )
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
  (forall (?p - drawer ?c - hexagonal_bin)
    (and
      (game-conserved
        (agent_holds ?c ?c)
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - pillow)
      (and
        (preference preferenceA
          (exists (?g - (either cube_block golfball))
            (exists (?k - (either golfball yellow_cube_block))
              (exists (?l - cube_block)
                (exists (?h - (either hexagonal_bin golfball key_chain))
                  (exists (?j - hexagonal_bin ?q - green_triangular_ramp ?o - chair ?e - blue_cube_block)
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
  (exists (?m - golfball)
    (forall (?b - game_object)
      (game-conserved
        (agent_holds ?b ?b)
      )
    )
  )
)
(:constraints
  (and
    (forall (?n - dodgeball)
      (and
        (preference preferenceA
          (exists (?j - tall_cylindrical_block ?q ?t - dodgeball)
            (then
              (once (and (in_motion ?q ?q) (agent_holds front) ) )
              (hold-to-end (not (in_motion ?n ?t) ) )
              (once (not (and (exists (?i ?o - block) (agent_holds ?t) ) (and (not (in_motion rug) ) (and (in_motion agent ?n) (adjacent ?q floor) ) ) ) ) )
            )
          )
        )
        (preference preferenceB
          (exists (?a - hexagonal_bin)
            (exists (?g - dodgeball ?t ?d ?z ?s ?b ?g - hexagonal_bin)
              (at-end
                (and
                  (in_motion sideways ?s)
                  (in_motion ?n ?a)
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?z - ball ?r - cube_block)
        (exists (?i - (either triangular_ramp))
          (then
            (once (same_color agent) )
            (once (agent_holds ?i) )
            (once (in_motion ?r) )
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
      (exists (?l - chair ?e - (either pyramid_block book laptop))
        (forall (?w - cube_block ?j - teddy_bear)
          (and
            (game-conserved
              (and
                (agent_holds ?j)
                (agent_holds ?j ?e)
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
    (forall (?l - dodgeball)
      (forall (?w - hexagonal_bin)
        (and
          (and
            (game-optional
              (agent_holds ?l ?l)
            )
            (forall (?a - cube_block)
              (and
                (and
                  (and
                    (game-optional
                      (< 1 10)
                    )
                  )
                  (and
                    (exists (?i - doggie_bed ?n - block ?k - hexagonal_bin)
                      (game-conserved
                        (< 2 (distance 5 agent))
                      )
                    )
                    (exists (?g - shelf)
                      (exists (?n - hexagonal_bin)
                        (and
                          (and
                            (game-conserved
                              (agent_holds ?l)
                            )
                          )
                        )
                      )
                    )
                  )
                  (game-conserved
                    (in_motion ?a)
                  )
                )
                (and
                  (game-optional
                    (not
                      (touch ?w)
                    )
                  )
                  (forall (?c - hexagonal_bin)
                    (exists (?e - cube_block ?r ?m ?t - cube_block)
                      (and
                        (exists (?s - game_object)
                          (forall (?y - game_object)
                            (and
                              (and
                                (and
                                  (game-optional
                                    (in_motion ?s)
                                  )
                                  (game-conserved
                                    (agent_holds ?a)
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
                  (in ?l)
                )
              )
            )
            (game-conserved
              (not
                (touch ?l)
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
      (exists (?n - game_object)
        (exists (?m - hexagonal_bin ?m - hexagonal_bin ?o - cube_block ?c - building)
          (then
            (any)
            (hold (agent_holds ?n) )
            (once (= (distance ?n ?n) (distance ?c ?c)) )
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
      (forall (?k - (either basketball key_chain) ?f - chair ?h - (either cylindrical_block golfball teddy_bear))
        (at-end
          (in_motion ?h)
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
  (exists (?f - dodgeball ?s - cube_block)
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
      (exists (?a - doggie_bed)
        (exists (?r - cube_block)
          (exists (?p - pillow)
            (then
              (once (in ?p) )
              (hold (on ?r) )
              (once (not (in_motion ?p ?a) ) )
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
    (forall (?f - (either lamp triangle_block tall_cylindrical_block) ?u - wall ?s - (either dodgeball laptop) ?z - block ?l - cube_block)
      (game-conserved
        (in ?l ?l)
      )
    )
    (exists (?l - (either dodgeball book))
      (and
        (game-optional
          (on ?l)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?r - golfball)
        (exists (?b - ball)
          (at-end
            (on ?b)
          )
        )
      )
    )
    (preference preferenceB
      (exists (?q - golfball)
        (exists (?m ?t ?p - triangular_ramp)
          (exists (?u ?n - cube_block ?z - hexagonal_bin ?e - ball ?n - cube_block)
            (at-end
              (agent_holds ?q yellow)
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
    (forall (?z - desk_shelf)
      (and
        (preference preferenceA
          (then
            (once (in bed) )
            (hold-to-end (and (adjacent rug) (not (not (and (exists (?s ?v - cube_block ?v - hexagonal_bin) (in ?v) ) (on ?z ?z) (and (agent_holds ?z rug) (agent_holds ?z) ) ) ) ) (< (distance 10 ?z) (distance agent)) ) )
            (once (in_motion ?z ?z) )
          )
        )
        (preference preferenceB
          (exists (?j - cube_block ?w - dodgeball)
            (exists (?j - hexagonal_bin ?b - building ?a - hexagonal_bin ?e - hexagonal_bin)
              (then
                (once (and (= 1 (distance ?w 10)) (in_motion ?w ?z) (and (and (not (in_motion ?z) ) (not (agent_holds ?w) ) ) (on ?w) ) ) )
                (once (or (not (adjacent ?z) ) (adjacent brown) ) )
                (once (and (adjacent_side ?z ?z) (<= 6 1) ) )
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
    (exists (?j ?n ?d - tan_cube_block ?t - curved_wooden_ramp)
      (exists (?k - dodgeball)
        (forall (?a - curved_wooden_ramp ?o - (either mug flat_block dodgeball))
          (and
            (and
              (game-optional
                (and
                  (and
                    (in_motion top_shelf ?t)
                    (in_motion ?k ?o)
                  )
                  (in_motion ?k ?t)
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
    (forall (?q - ball ?p ?t - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?d - building ?b - building ?z - dodgeball)
            (exists (?y - triangular_ramp ?w ?e ?r - chair ?g - (either main_light_switch alarm_clock credit_card blue_cube_block))
              (exists (?u - (either bridge_block book))
                (exists (?k - hexagonal_bin ?r - doggie_bed)
                  (exists (?f - game_object ?w - hexagonal_bin)
                    (exists (?o - hexagonal_bin)
                      (exists (?m ?h ?s - shelf)
                        (then
                          (hold-to-end (agent_holds ?g) )
                          (hold-while (between ?r) (toggled_on agent) )
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
        (forall (?p - block)
          (and
            (forall (?m - game_object)
              (exists (?b - block ?j - building)
                (game-conserved
                  (not
                    (agent_holds ?j)
                  )
                )
              )
            )
            (forall (?s - block)
              (exists (?j - chair ?v - ball ?o - blinds)
                (game-conserved
                  (in ?p)
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
                  (exists (?b - cube_block)
                    (toggled_on ?b)
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
    (forall (?u - dodgeball)
      (and
        (preference preferenceA
          (exists (?j - block)
            (exists (?k - dodgeball)
              (exists (?m - hexagonal_bin ?f - dodgeball)
                (exists (?x - hexagonal_bin ?e - bridge_block)
                  (exists (?x - (either dodgeball golfball pyramid_block))
                    (exists (?m - (either pyramid_block basketball dodgeball))
                      (exists (?w - game_object)
                        (exists (?q - dodgeball)
                          (then
                            (hold (on ?x) )
                            (once (in_motion ?u) )
                            (once (in_motion bed ?e ?w) )
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
            (once (and (not (in_motion ?u ?u bed) ) (in_motion bed) (in_motion ?u) (exists (?p - (either triangle_block pyramid_block)) (not (in ?p ?u) ) ) (in rug desk) (and (on ?u ?u) (in_motion ?u) ) ) )
            (once (not (agent_holds ?u) ) )
            (once (adjacent_side ?u) )
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
        (once (exists (?h - hexagonal_bin) (in ?h) ) )
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
    (forall (?t - building ?h - block)
      (and
        (not
          (and
            (game-conserved
              (in_motion ?h ?h)
            )
          )
        )
        (exists (?l - game_object ?a ?j - wall)
          (game-conserved
            (not
              (adjacent ?a)
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
        (once (and (in_motion ?xxx) (in_motion ?xxx) (and (exists (?i - (either yellow_cube_block pencil) ?f - blinds) (and (agent_holds ?f) (not (and (and (and (< 1 1) ) (not (agent_holds ?f ?f) ) ) (adjacent ?f) ) ) ) ) (in_motion ?xxx) ) (adjacent_side ?xxx) (= (distance ) (distance ?xxx 8)) (< (distance door ?xxx ?xxx) (distance ?xxx ?xxx)) (same_color ?xxx) (and (= 1 7) (on bed ?xxx) ) ) )
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
  (forall (?t - hexagonal_bin)
    (exists (?s ?j - bridge_block)
      (and
        (game-optional
          (in_motion ?s)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - hexagonal_bin ?f - chair)
      (and
        (preference preferenceA
          (then
            (hold-for 8 (and (< (distance ?f 2) 4) (agent_holds ?f) ) )
            (once (in_motion ?f ?f) )
            (once (agent_holds ?f) )
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
      (exists (?g - cube_block)
        (game-optional
          (open ?g ?g)
        )
      )
      (and
        (game-conserved
          (and
            (in_motion ?xxx ?xxx)
            (forall (?x - red_dodgeball)
              (agent_holds east_sliding_door)
            )
          )
        )
      )
      (forall (?k - (either golfball cellphone))
        (game-conserved
          (agent_holds ?k ?k)
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
      (exists (?k - block ?b - cube_block)
        (then
          (hold (in_motion ?b) )
          (hold (not (agent_holds ?b) ) )
          (hold-to-end (and (agent_holds back green_golfball ?b) (on color rug) ) )
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
  (exists (?h - hexagonal_bin)
    (or
      (exists (?x - cube_block ?j - hexagonal_bin)
        (game-optional
          (agent_holds agent ?h)
        )
      )
      (game-conserved
        (touch ?h agent)
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
    (forall (?j - hexagonal_bin ?z ?l ?s ?j ?w ?e - hexagonal_bin)
      (and
        (preference preferenceB
          (then
            (hold (in_motion ?j) )
            (once (adjacent pink_dodgeball) )
            (once (touch ?l ?l) )
          )
        )
        (preference preferenceC
          (then
            (hold (on ?s ?s) )
            (once (agent_holds ?z) )
            (hold-while (not (in_motion ?w) ) (on ?e) )
          )
        )
        (preference preferenceD
          (exists (?k - ball)
            (then
              (once (< 0.5 0) )
              (once (and (touch ?z ?e) (in_motion ?w) ) )
              (hold-while (and (not (and (not (touch ?s) ) (touch ?k) ) ) (not (touch ?w) ) ) (and (< (distance ?s 9) 1) (agent_holds bed) ) )
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
  (forall (?k - hexagonal_bin)
    (game-conserved
      (and
        (and
          (and
            (< 2 (distance desk ?k))
          )
          (and
            (agent_holds ?k bed)
          )
        )
        (< (distance ?k 0) (distance agent ?k ?k))
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?a - doggie_bed)
        (exists (?g - curved_wooden_ramp)
          (exists (?j - hexagonal_bin ?u - hexagonal_bin ?j - game_object)
            (then
              (hold (and (touch ?a) (in_motion ?g) ) )
              (hold (not (agent_holds desk ?j) ) )
              (hold-while (on upright ?a) (not (not (and (adjacent ?a bridge_block) (not (not (or (and (exists (?y - (either cylindrical_block teddy_bear)) (and (not (agent_holds ?a) ) (touch ?y) (in ?j agent) ) ) (and (or (in ?a ?a) (in_motion ?a) ) (in_motion pink) ) (on desk) (on ?a) ) (in ?g) (in_motion ?j desk) (not (and (in_motion rug floor) (in_motion ?a) (not (agent_holds ?g ?j ?j ?j) ) ) ) ) ) ) (and (in_motion ?j brown) (agent_holds ?g) ) ) ) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?a - game_object)
        (then
          (hold (and (on ?a ?a) (adjacent ?a) ) )
          (once (adjacent ?a rug) )
          (once (not (in_motion ?a ?a) ) )
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
        (forall-sequence (?u - red_pyramid_block)
          (then
            (once (in ?u agent) )
            (once (not (and (in ) (agent_holds ?u) ) ) )
            (hold-while (agent_holds ?u) (not (in_motion ?u) ) )
          )
        )
        (once (object_orientation ?xxx) )
        (once (in_motion ?xxx) )
      )
    )
    (preference preferenceB
      (exists (?h - (either dodgeball floor))
        (exists (?s - (either dodgeball tall_cylindrical_block) ?a - (either cellphone top_drawer) ?k - (either book floor) ?j ?p - doggie_bed ?u - teddy_bear)
          (exists (?l - (either yellow_cube_block cellphone))
            (exists (?i - hexagonal_bin)
              (then
                (hold-while (not (< 1 1) ) (not (not (touch ?l ?h) ) ) )
                (hold (agent_holds ?u) )
                (hold (not (game_start ?l rug) ) )
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
        (once (and (and (and (in_motion ?xxx) (and (adjacent floor ?xxx) (not (and (not (and (not (< (distance side_table ?xxx) 6) ) (not (not (and (not (on ?xxx ?xxx) ) (not (and (on floor ?xxx) (in_motion ?xxx) (and (and (forall (?s - doggie_bed) (in_motion ?s pink_dodgeball) ) (on ?xxx ?xxx) (agent_holds ?xxx) ) (in_motion ?xxx) ) (not (agent_holds ?xxx ?xxx ?xxx) ) ) ) (in_motion ?xxx ?xxx) ) ) ) ) ) (agent_holds ?xxx) ) ) (in_motion ?xxx agent) (not (exists (?f - hexagonal_bin ?s - (either blue_cube_block ball)) (in_motion ?s ?s) ) ) (adjacent_side ?xxx ?xxx) (and (agent_holds ?xxx) (exists (?h - bridge_block ?j - watch) (in_motion agent ?j) ) (on ?xxx) ) ) (and (adjacent ?xxx) (< 2 (distance ?xxx room_center)) ) (on ?xxx bridge_block) ) (and (on ?xxx) (touch ?xxx door) ) ) (agent_holds ?xxx) ) )
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
  (forall (?z - cube_block)
    (game-conserved
      (in_motion ?z west_wall)
    )
  )
)
(:constraints
  (and
    (forall (?s - hexagonal_bin ?a - curved_wooden_ramp)
      (and
        (preference preferenceA
          (then
            (once (forall (?y - doggie_bed) (in_motion ?y ?y) ) )
            (once (agent_holds agent) )
            (hold (in_motion ?a ?a) )
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
    (exists (?c - hexagonal_bin)
      (game-conserved
        (on ?c ?c)
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
    (forall (?v ?s - color)
      (and
        (preference preferenceA
          (exists (?f ?i - triangular_ramp ?m - curved_wooden_ramp)
            (then
              (once (and (exists (?x - (either cube_block)) (and (not (agent_holds ?v) ) (and (exists (?t - block) (equal_z_position ?s) ) (not (toggled_on desk) ) (in_motion ?x ?m) (in_motion ?s ?m) (type ?v ?s) (on ?v) (and (not (and (not (same_type ?s) ) (and (and (not (>= 6 (distance ?s ?s)) ) (is_setup_object rug) (or (on ?m) (in ?s) ) (and (and (in_motion ?x pink) (adjacent_side ?s) ) (not (not (on ?v) ) ) ) (in_motion ?s) (object_orientation ?x) ) (and (in_motion ?m) (in ?v) ) ) ) ) (not (same_object ?s ?m) ) ) (not (and (not (agent_holds ?m) ) (agent_holds ?x) ) ) ) ) ) (< 1 7) ) )
              (hold (agent_holds ?v) )
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
    (forall (?p - ball)
      (and
        (preference preferenceA
          (at-end
            (agent_holds )
          )
        )
        (preference preferenceB
          (exists (?r - pillow ?d ?a - dodgeball)
            (then
              (once (touch ?p) )
              (once (and (not (> (distance ?d ?d) 1) ) (not (not (agent_holds ?d ?d) ) ) ) )
              (once (agent_holds ?d ?a) )
            )
          )
        )
        (preference preferenceC
          (then
            (hold (in ?p ?p) )
            (once (not (rug_color_under ?p ?p) ) )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?w - chair ?k - building ?k - cube_block)
        (at-end
          (not
            (agent_holds ?k)
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
      (exists (?l - building)
        (then
          (hold (forall (?y - cube_block ?g - hexagonal_bin ?u - (either golfball dodgeball)) (and (in_motion ?l) (agent_holds ?l ?l) ) ) )
          (once (touch pink_dodgeball ?l) )
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
  (exists (?w ?v - dodgeball)
    (game-conserved
      (on ?v)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?k - cube_block)
        (exists (?s - hexagonal_bin ?v - block)
          (then
            (once (and (agent_holds ?v) (not (and (in ?v ?v) (on desk ?v) (same_color ?k) (agent_holds ?k ?k) ) ) ) )
            (once (not (not (and (on ?k) (and (and (not (< 1 (distance ?v desk)) ) (on ?v ?v) ) (not (game_start desk) ) ) ) ) ) )
            (once (not (in_motion ?v ?k) ) )
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
  (exists (?r - dodgeball)
    (game-conserved
      (agent_holds ?r ?r)
    )
  )
)
(:constraints
  (and
    (forall (?j ?r - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?z - ball)
            (then
              (once (and (on pillow ?r) (touch ?z rug) ) )
              (once (agent_holds ?r) )
            )
          )
        )
      )
    )
    (forall (?n - tall_cylindrical_block)
      (and
        (preference preferenceB
          (exists (?b - (either dodgeball golfball) ?r - cube_block)
            (then
              (hold (agent_holds ?n) )
              (once (agent_holds ?n) )
              (hold (on ?n) )
              (hold (on ?r) )
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
      (exists (?g - wall ?a - ball)
        (exists (?m ?d - hexagonal_bin ?l - (either tall_cylindrical_block book golfball blue_cube_block dodgeball floor key_chain))
          (game-conserved
            (not
              (in_motion bed ?a)
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
      (exists (?g - game_object ?l - shelf)
        (exists (?h - cube_block)
          (exists (?w - ball)
            (at-end
              (in_motion ?h ?w)
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
  (exists (?m - hexagonal_bin ?b - triangular_ramp)
    (forall (?j - triangular_ramp ?i - (either rug rug))
      (exists (?c - (either yellow yellow))
        (game-conserved
          (agent_holds ?i ?c)
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
    (forall (?n - (either cube_block golfball cylindrical_block game_object blue_cube_block dodgeball blue_cube_block))
      (and
        (preference preferenceB
          (then
            (once (agent_holds ?n ?n) )
            (hold (agent_holds ?n) )
            (hold (and (in_motion floor) (rug_color_under bed ?n) ) )
          )
        )
        (preference preferenceC
          (exists (?d - game_object)
            (exists (?z - (either pillow pyramid_block))
              (exists (?w - beachball)
                (exists (?y - building)
                  (then
                    (once (not (not (and (on ?y ?z) ) ) ) )
                    (once (and (agent_holds ?d) (in_motion ?w) ) )
                    (once (not (not (not (> (distance desk 0 ?y) (distance ?w green_golfball)) ) ) ) )
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
      (exists (?o - sliding_door)
        (then
          (hold (in_motion ?o) )
          (once (adjacent ?o ?o) )
          (once (and (and (same_color ?o south_west_corner) (is_setup_object ?o ?o) (= (distance room_center ?o) (distance ?o) 1 2) ) (agent_holds ?o) ) )
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
  (exists (?r - game_object)
    (not
      (and
        (and
          (game-conserved
            (in ?r)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?p - dodgeball)
      (and
        (preference preferenceA
          (exists (?f - game_object)
            (then
              (hold-while (is_setup_object ?f) (not (not (< (distance room_center ?p) (distance front_left_corner agent)) ) ) )
              (once (in_motion ?p) )
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
      (exists (?a - triangular_ramp)
        (exists (?j - hexagonal_bin)
          (then
            (once (and (touch ?a) (and (not (in_motion ?a floor) ) (in ?j agent) (and (and (agent_holds ?j ?a) (in_motion ?j) ) (in_motion ?j) (and (in_motion ?j) (not (agent_holds ?a) ) (on ?a ?a) (and (faces ?j ?a) (and (not (not (on ?j) ) ) (not (on ?a ?j) ) ) ) (in_motion ?j) ) (exists (?s - color) (in ?j) ) ) ) (and (agent_holds ?j ?j) (in ?j ?j) ) (on ?a) ) )
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
      (exists (?b - teddy_bear)
        (exists (?c - cube_block)
          (then
            (hold (not (not (in ?b) ) ) )
            (hold-while (forall (?r - hexagonal_bin) (in_motion ?c ?c) ) (not (in_motion ?b) ) (and (on ?c) (and (or (and (not (in_motion ?b agent) ) (and (and (not (agent_holds ?c) ) (in_motion front) ) (not (agent_holds ?c) ) ) ) (on ?b ?c ?b) ) (in_motion ?b) ) ) )
            (hold (not (and (agent_holds ?b) (and (and (agent_holds ?b ?c) (not (in_motion ?b) ) ) (and (and (not (in_motion ?c floor) ) (not (in_motion ?c ?c) ) ) (on ?b) ) ) ) ) )
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
      (exists (?y - teddy_bear ?s - color)
        (exists (?t - (either pyramid_block pink pyramid_block wall wall flat_block ball) ?z - ball)
          (exists (?v - hexagonal_bin ?e - hexagonal_bin)
            (game-conserved
              (in_motion ?e ?e)
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
      (exists (?m ?c - ball ?g - (either mug cellphone golfball) ?b - hexagonal_bin)
        (then
          (hold (agent_holds ?b) )
          (once (and (and (forall (?v ?o - rug) (not (in ?v desk) ) ) (on bed) ) (in_motion desk ?b) ) )
          (hold (touch ?b) )
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
    (forall (?q - block)
      (and
        (preference preferenceB
          (exists (?j - color)
            (exists (?a - ball)
              (exists (?w - shelf)
                (then
                  (once (on ?w) )
                  (once (and (and (not (not (and (and (adjacent ?j) (in_motion ) ) (exists (?h - ball) (in_motion ?j agent) ) ) ) ) (not (exists (?z - dodgeball) (and (and (faces bed) (agent_holds ?q bed) ) (agent_holds ?a door) (agent_holds ?z) ) ) ) ) (on ?w) ) )
                  (hold (in_motion desk) )
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?e - hexagonal_bin ?h - doggie_bed)
        (exists (?w - dodgeball ?v - chair)
          (then
            (once (not (and (same_type ?v upside_down) (not (in_motion upright) ) ) ) )
            (once (same_color ?h ?h) )
            (hold (in_motion ?h) )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?z - hexagonal_bin)
        (then
          (once (agent_holds ?z ?z) )
          (once (in_motion ?z) )
          (once (in_motion ?z) )
        )
      )
    )
    (preference preferenceE
      (exists (?w - cylindrical_block ?j - dodgeball)
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
  (exists (?m - desk_shelf)
    (game-optional
      (on bed agent)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?j - hexagonal_bin)
        (exists (?k - game_object ?o - (either cube_block cube_block) ?o - hexagonal_bin ?h - triangular_ramp)
          (exists (?b - color ?e - hexagonal_bin)
            (then
              (once (not (object_orientation ?h ?e) ) )
              (once (in ?h) )
              (once (not (equal_z_position ?j) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?s - wall)
        (exists (?r - dodgeball)
          (then
            (hold (on ?s ?r) )
            (once (and (and (in ?r) (not (in_motion ?r) ) ) (in_motion ?r ?s) ) )
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
      (exists (?m - block ?m - pyramid_block)
        (then
          (hold-while (on ?m) (in_motion bed) )
          (hold (not (not (in_motion desk) ) ) )
          (once (or (not (touch ?m) ) (on ?m) (and (adjacent ?m) (and (above desk) (not (or (not (and (agent_holds ?m) (and (agent_holds ?m ?m) (and (or (not (and (on ?m ?m) (and (agent_holds ?m ?m) (and (and (or (on ?m ?m) (touch ?m) ) (> 2 1) ) (not (in_motion ?m ?m) ) ) (agent_holds ?m) ) ) ) (in_motion ?m) ) (not (agent_holds ?m ?m) ) (not (faces ?m) ) ) ) ) ) (not (agent_holds ?m ?m) ) ) ) ) ) ) )
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
    (forall (?o - beachball ?u - teddy_bear)
      (and
        (preference preferenceA
          (exists (?r - dodgeball)
            (exists (?t - teddy_bear)
              (forall (?d - (either floor pencil dodgeball))
                (then
                  (once-measure (adjacent ?d ?r) (distance ?t 7) )
                  (once (and (agent_holds ?t) (and (on ?r) (not (and (agent_holds ?d) (agent_holds ?t) ) ) ) ) )
                  (once (not (not (on top_drawer ?t) ) ) )
                )
              )
            )
          )
        )
        (preference preferenceB
          (then
            (hold-while (in_motion ?u ?u) (on ?u ?u) )
            (once-measure (agent_holds ?u desk) (distance ?u desk) )
            (once (agent_holds ?u ?u) )
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
  (exists (?y - golfball)
    (game-optional
      (in_motion ?y ?y)
    )
  )
)
(:constraints
  (and
    (forall (?b - color)
      (and
        (preference preferenceA
          (exists (?m - (either tall_cylindrical_block alarm_clock))
            (then
              (hold (not (not (not (touch ?m agent) ) ) ) )
              (once (agent_holds ?m ?b) )
              (once (agent_holds ?b ?b) )
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
  (exists (?i - ball)
    (and
      (exists (?l - hexagonal_bin)
        (and
          (game-conserved
            (on ?i ?l)
          )
          (game-conserved
            (faces ?l)
          )
          (and
            (and
              (or
                (game-conserved
                  (adjacent ?l)
                )
                (game-conserved
                  (in_motion agent)
                )
                (game-conserved
                  (on ?i ?i)
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
  (exists (?m - (either main_light_switch cube_block))
    (forall (?x - hexagonal_bin)
      (game-optional
        (not
          (not
            (in ?x)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?d - block)
        (at-end
          (not
            (not
              (and
                (agent_holds ?d ?d)
                (>= (distance front) 2)
                (in ?d)
                (not
                  (not
                    (in_motion ?d ?d)
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
    (forall (?q - (either basketball dodgeball))
      (and
        (preference preferenceA
          (then
            (once (not (not (and (in ?q ?q) (and (and (agent_holds ?q ?q) (and (and (in_motion ?q) (in_motion ?q brown) ) (in_motion ?q) ) ) (agent_holds ?q) ) ) ) ) )
            (once (not (in_motion ?q ?q) ) )
            (once (agent_holds ?q) )
          )
        )
      )
    )
    (forall (?x - shelf ?y ?v - cylindrical_block)
      (and
        (preference preferenceB
          (then
            (forall-sequence (?h - game_object)
              (then
                (once (in ?y) )
                (once (and (and (not (in_motion ?v) ) (touch ?y) ) (in ?v) ) )
                (hold (not (in south_west_corner) ) )
              )
            )
            (once (or (agent_holds desk ?y) (not (exists (?o - teddy_bear ?j - building) (agent_holds ?j desktop) ) ) (adjacent ?y) (not (agent_holds ?v) ) ) )
            (hold (agent_holds ?y) )
          )
        )
        (preference preferenceC
          (then
            (once (exists (?q - game_object) (agent_holds ?q ?y) ) )
            (hold (agent_holds ?y ?v) )
            (once (agent_holds ?v) )
          )
        )
        (preference preferenceD
          (then
            (once (same_color ?v floor) )
            (once (object_orientation ?y) )
            (once (agent_holds ?v) )
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
      (exists (?g - game_object)
        (then
          (hold (not (and (agent_holds ?g ?g) (touch ?g ?g) ) ) )
          (once (agent_holds ?g ?g) )
          (once (and (in ?g ?g) (in_motion ?g) (on ?g ?g) ) )
          (hold (and (same_color ?g ?g) (in ?g) ) )
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
    (forall (?n - pyramid_block)
      (in_motion ?n agent)
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
                (forall (?d - drawer)
                  (agent_holds ?d)
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
          (exists (?m - dodgeball)
            (in_motion ?m ?m)
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
  (forall (?t - hexagonal_bin ?l - hexagonal_bin)
    (and
      (forall (?q - hexagonal_bin)
        (exists (?g - (either game_object dodgeball))
          (game-conserved
            (not
              (faces ?g)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?t - hexagonal_bin ?v - game_object)
      (and
        (preference preferenceA
          (then
            (hold (not (not (agent_holds ?v ?v) ) ) )
            (once (not (and (and (in_motion ?v ?v) (not (not (type ?v ?v) ) ) ) (in_motion front ?v) ) ) )
            (hold-while (agent_holds ?v agent) (in_motion ?v) )
          )
        )
        (preference preferenceB
          (exists (?l - game_object)
            (then
              (hold-while (agent_holds main_light_switch) (not (agent_holds ?l ?l) ) )
              (once (in_motion ?v ?v) )
              (hold (in_motion ?l) )
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
      (exists (?n ?m ?i - cube_block ?a - hexagonal_bin)
        (forall (?r ?o - ball)
          (game-conserved
            (and
              (open ?r ?o)
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
  (exists (?f - dodgeball)
    (game-optional
      (> (distance ?f) (distance ?f desk))
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
  (forall (?t - (either basketball ball book cylindrical_block key_chain cellphone pen) ?n - (either laptop laptop) ?v - dodgeball)
    (game-optional
      (agent_holds ?v)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?e - ball)
        (exists (?k - hexagonal_bin)
          (then
            (hold (and (and (broken ?e ?e) (on ?k) ) ) )
            (once (touch ?k) )
            (once (and (not (not (on ?e ?e) ) ) (and (not (in_motion ?e ?e) ) (and (and (and (exists (?f - game_object ?v - (either basketball basketball)) (and (not (and (on ?e ?k) (agent_holds ?v ?e) (in_motion ?e ?v) ) ) (in_motion ?k top_shelf) ) ) (not (same_object ?e ?e) ) (agent_holds pillow) ) (on ?k ?e) ) (in ?k upright) ) (adjacent ?k) ) (not (and (not (and (not (in ?e ?e) ) (and (in_motion ?k) (< 4 1) ) ) ) (and (adjacent ?e ?k) (in_motion ?k) ) ) ) ) )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?w ?q - hexagonal_bin)
        (then
          (once (agent_holds rug ?w) )
          (hold (= 1 (distance ?w ?w) (distance_side ?w)) )
          (hold-while (and (< 2 (distance room_center agent)) (not (not (in_motion ?w ?w) ) ) ) (in_motion ?w ?q ?w) (touch ?w) )
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
  (exists (?y - chair)
    (game-conserved
      (not
        (not
          (in_motion ?y)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?u - game_object)
        (then
          (once (in_motion ?u) )
          (hold (in_motion agent) )
          (once (type ?u) )
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
  (forall (?l - hexagonal_bin)
    (game-conserved
      (and
        (< (distance front 8 ?l) (distance ?l room_center))
        (not
          (< (distance ?l ?l) 5)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?e - hexagonal_bin)
        (then
          (once (and (on ?e door) (and (same_color ?e) (in_motion desk) (and (in_motion agent) (not (adjacent_side ?e ?e) ) ) ) ) )
          (once (not (in ?e) ) )
          (hold-while (exists (?r - curved_wooden_ramp ?b ?u - (either chair cube_block book red)) (and (touch ?b) (and (and (in_motion ?e) (and (agent_holds ?u ?u) (in ?b) ) ) (agent_holds ?b ?e) ) (not (agent_holds ?e) ) ) ) (in ?e ?e) )
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
      (exists (?a - (either blue_cube_block pyramid_block))
        (exists (?p - curved_wooden_ramp)
          (then
            (once (in_motion tan) )
            (once (not (agent_holds ?p ?a) ) )
            (once (or (agent_holds ?a ?p) (not (and (and (and (not (agent_holds ?a ?p) ) (not (and (agent_holds ?p ?a) (on ?a ?a) ) ) ) (not (adjacent ?p) ) ) (in_motion ?a ?a) ) ) (not (and (> 5 1) (and (agent_holds ?p ?p) (and (and (agent_holds bed ?a) (not (in_motion ?a ?p) ) (in_motion ) ) (exists (?b - curved_wooden_ramp ?c - (either dodgeball cylindrical_block)) (not (on ?p) ) ) ) ) (agent_holds ?p bed) ) ) (or (and (not (in agent ?a) ) (on ?p ?p) ) (agent_holds ?a ?p) ) ) )
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (not (agent_holds ?xxx) ) )
        (once (in_motion ?xxx) )
        (once (and (and (agent_holds ?xxx ?xxx) (not (not (not (not (not (forall (?b - (either teddy_bear golfball)) (adjacent ?b) ) ) ) ) ) ) ) (on ?xxx ?xxx) ) )
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
  (forall (?t - hexagonal_bin)
    (forall (?u - (either bridge_block mug basketball) ?n - dodgeball)
      (and
        (exists (?e - pyramid_block ?g - doggie_bed)
          (and
            (and
              (exists (?j - hexagonal_bin)
                (forall (?m - hexagonal_bin ?a - beachball)
                  (game-conserved
                    (not
                      (agent_holds brown)
                    )
                  )
                )
              )
              (game-conserved
                (or
                  (in_motion ?t ?t)
                )
              )
            )
            (and
              (or
                (and
                  (exists (?z - hexagonal_bin)
                    (forall (?u - book ?x - teddy_bear)
                      (and
                        (exists (?r - building)
                          (and
                            (game-optional
                              (exists (?f - cube_block)
                                (in_motion ?f ?x)
                              )
                            )
                            (forall (?y - golfball)
                              (exists (?k ?w - dodgeball ?i - (either desktop))
                                (game-conserved
                                  (exists (?c - hexagonal_bin ?s - hexagonal_bin ?s - (either dodgeball doggie_bed) ?q - cube_block)
                                    (in_motion ?g ?x)
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
                            (in_motion ?g ?x)
                          )
                        )
                        (and
                          (exists (?e - cube_block ?s - chair)
                            (exists (?r - hexagonal_bin ?w ?p - (either triangle_block))
                              (exists (?r - building)
                                (exists (?v - hexagonal_bin)
                                  (forall (?f - game_object)
                                    (game-optional
                                      (in_motion ?p)
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
                        (agent_holds ?n)
                        (on desk)
                        (not
                          (not
                            (not
                              (not
                                (and
                                  (agent_holds ?g)
                                  (agent_holds ?g)
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
          (agent_holds ?t)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - (either golfball blue_cube_block) ?j - dodgeball)
      (and
        (preference preferenceA
          (exists (?h - cube_block ?u - game_object)
            (exists (?i - hexagonal_bin ?b - ball ?i - hexagonal_bin)
              (then
                (once (or (and (adjacent ?i) (toggled_on ?j agent) ) (and (in ?i ?u) (< 7 (distance 3 ?u)) ) ) )
                (once (not (and (not (in_motion ?i) ) (not (not (and (agent_holds front) (on ?i) (and (between ?i ?i) (and (and (not (and (and (in ?u) (in_motion ?j) ) (in pink) (in_motion ?u ?j) (and (not (same_color ?u) ) ) ) ) (in_motion ?j) ) (adjacent ?i) ) (in_motion ?i) ) ) ) ) ) ) )
                (hold-while (agent_holds ?i ?u) (in_motion ?u) (and (> (distance_side agent ?i) 1) (in_motion ?j ?u) (and (not (agent_holds ?u) ) (not (< (distance ?i ?u) 1) ) ) ) )
                (hold-while (on ?i ?j) (and (in_motion ?i ?u) (in_motion ?j ?j) ) )
              )
            )
          )
        )
      )
    )
    (forall (?o - triangular_ramp)
      (and
        (preference preferenceB
          (exists (?t - hexagonal_bin ?z - hexagonal_bin)
            (at-end
              (agent_holds ?z)
            )
          )
        )
        (preference preferenceC
          (then
            (hold-while (in_motion agent ?o) (adjacent ?o) (and (in_motion ?o) (object_orientation ?o ?o) ) (in ?o ?o) )
            (hold (not (not (in desk ?o) ) ) )
            (hold (in ?o) )
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
  (exists (?l - red_dodgeball)
    (exists (?n - shelf)
      (and
        (and
          (exists (?f - dodgeball)
            (exists (?z - block ?i - color)
              (and
                (game-conserved
                  (< (distance ?l) (x_position 7 9))
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
    (forall (?o - block)
      (and
        (preference preferenceA
          (exists (?l - (either cube_block yellow_cube_block))
            (exists (?z - shelf ?x ?e ?s ?d ?a ?i - (either tall_cylindrical_block dodgeball dodgeball) ?q - curved_wooden_ramp)
              (then
                (once (exists (?m - (either cube_block yellow_cube_block)) (and (exists (?w - (either dodgeball yellow_cube_block)) (and (on ?o ?q) (and (agent_holds ?m) (agent_holds desk left) (not (on ?q) ) (or (not (adjacent ?l) ) (exists (?p - hexagonal_bin) (not (same_color ?l ?p) ) ) ) ) ) ) (in_motion ?o ?o) ) ) )
                (once (<= (distance ?o ?o) (distance bed)) )
                (once (agent_holds ?o) )
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
  (exists (?t - watch ?c - chair ?n - (either blue_cube_block basketball top_drawer))
    (game-conserved
      (in_motion ?n)
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

