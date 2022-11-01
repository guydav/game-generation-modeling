(define (game game-id) (:domain domain-name)
(:setup
  (exists (?d - (either dodgeball yellow))
    (or
      (or
        (game-conserved
          (in ?d pink)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - dodgeball)
      (and
        (preference preferenceA
          (then
            (once (touch ?i ?i) )
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
      (exists (?j - doggie_bed)
        (exists (?m - dodgeball)
          (exists (?u - hexagonal_bin)
            (then
              (once (and (<= 1 (distance agent bed)) (in brown) (in ?m) ) )
              (once (on ?m ?j) )
              (once (agent_holds ?j) )
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
  (count preferenceA:beachball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?d - flat_block)
    (and
      (and
        (and
          (game-conserved
            (not
              (in_motion ?d)
            )
          )
        )
      )
      (forall (?k - shelf)
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
      (forall (?u - teddy_bear)
        (and
          (and
            (game-conserved
              (not
                (in_motion ?u)
              )
            )
            (forall (?c ?y ?z - ball)
              (exists (?e - sliding_door)
                (and
                  (exists (?v - (either laptop pillow) ?s - bridge_block)
                    (and
                      (exists (?l - building)
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
    (forall (?u - block)
      (and
        (preference preferenceA
          (exists (?g - chair)
            (then
              (hold (not (in ?g agent) ) )
              (once (in_motion ?g ?g) )
              (once (agent_holds ?u) )
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
          (<= 10 (count preferenceA:yellow_cube_block) )
        )
        (or
          (> (count-shortest preferenceA:blue_dodgeball) (* (count-once-per-external-objects preferenceA:blue_dodgeball) (* (count preferenceA:green) (count preferenceA:hexagonal_bin) )
              (+ (* 1 2 )
                5
                (count preferenceA:yellow_cube_block:yellow)
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
                  (exists (?g - game_object ?t - ball)
                    (and
                      (or
                        (in ?t ?t)
                        (agent_holds agent)
                      )
                      (<= (distance ?t bed ?t) 5)
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
      (exists (?i - pillow)
        (at-end
          (and
            (is_setup_object ?i)
            (in_motion ?i)
            (on upside_down)
          )
        )
      )
    )
    (forall (?k - dodgeball)
      (and
        (preference preferenceB
          (then
            (once (in_motion ?k ?k) )
            (once (and (not (agent_holds agent) ) (not (agent_holds ?k ?k) ) (in agent) ) )
            (hold (and (in ?k) (agent_holds ?k) (agent_holds ?k) (= (distance bed ?k)) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ 9 (* (- (>= 10 30 )
        )
        (< (count preferenceA:pink_dodgeball:purple) 4 )
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
    (forall (?o - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?l - (either golfball rug tall_cylindrical_block))
            (exists (?q - hexagonal_bin ?c - hexagonal_bin)
              (then
                (once (in_motion ?c) )
                (once (and (not (not (and (and (adjacent ?l front) (not (not (in_motion ?o) ) ) (agent_holds ?o) ) (touch ?l pink_dodgeball) ) ) ) (in_motion ?c ?c) (agent_holds floor) ) )
                (once (and (on ?l ?l) (and (in ?c) (not (and (in_motion ?c ?c ?o) (on right ?o) (and (and (not (agent_holds ?l) ) (not (adjacent ?o) ) ) (on ?l) (not (agent_holds ?l ?o) ) (in ?l) ) (and (= 6 4 1) (and (same_object ?l ?c) (not (and (in ?l ?o) (touch ?o) (in desk ?l) (above ?l ?c) ) ) ) ) (touch ?l) (and (agent_holds ?l) (in_motion ?c) ) (agent_holds ?l) ) ) (and (on ?o ?c) (agent_holds ?o) ) (< 1 (distance 1 ?o)) ) ) )
              )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?t - dodgeball)
        (then
          (once (and (agent_holds ?t rug) (touch ?t) ) )
          (once (agent_holds ?t) )
          (hold (faces bed right) )
        )
      )
    )
  )
)
(:terminal
  (>= 4 20 )
)
(:scoring
  (+ 1 (count-longest preferenceB:bed) (<= (* (count-once-per-objects preferenceB:golfball:golfball) (+ 5 (count preferenceB:yellow) )
        (count preferenceA:beachball)
      )
      (* (+ (+ (count preferenceA:yellow) 4 )
        )
        3
        (count preferenceB:pink)
        (* 4 )
        (* (count-total preferenceA:tall_cylindrical_block) (- (count-measure preferenceA:cube_block) )
        )
        (count preferenceB:red:dodgeball:pink)
      )
    )
    (* (count-once-per-objects preferenceA:block) 3 )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?h - red_dodgeball)
    (game-conserved
      (on ?h agent)
    )
  )
)
(:constraints
  (and
    (forall (?b ?u ?j - (either cylindrical_block cube_block key_chain))
      (and
        (preference preferenceA
          (then
            (once (and (and (exists (?q - cube_block ?p - hexagonal_bin) (not (adjacent agent ?b) ) ) (and (adjacent ?u) (touch ?b) ) ) (in_motion agent ?j) ) )
            (once (in_motion ?j) )
            (hold (in ?b) )
          )
        )
        (preference preferenceB
          (exists (?a - teddy_bear)
            (exists (?q - hexagonal_bin)
              (exists (?e - red_dodgeball)
                (exists (?w - shelf ?x - building)
                  (exists (?d - (either blue_cube_block golfball) ?o - (either yellow_cube_block))
                    (exists (?w - teddy_bear)
                      (then
                        (once (agent_holds ?e) )
                        (once (< 0.5 1) )
                        (hold-for 10 (in_motion bed ?u) )
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
      (exists (?c - cube_block)
        (then
          (once (and (agent_holds ?c floor) (not (same_object ?c) ) ) )
          (once (and (and (and (not (in_motion bed ?c) ) (agent_holds top_shelf) ) (and (not (touch ?c front_left_corner) ) (<= (distance agent room_center agent) 2) ) ) (in_motion ?c) ) )
          (hold (and (< (distance ?c 4) 1) (on ?c) (in ?c) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (and (exists (?h - beachball) (agent_holds ?h) ) (in_motion rug ?xxx) (<= 10 (distance_side 9 agent)) (not (in sideways ?xxx) ) ) )
        (hold (and (agent_holds ?xxx) (< (distance 5 ?xxx) 0.5) ) )
        (once (touch ?xxx) )
      )
    )
    (forall (?c - ball)
      (and
        (preference preferenceC
          (exists (?t - (either laptop hexagonal_bin))
            (exists (?p - shelf)
              (then
                (once (exists (?q - (either cellphone blue_cube_block dodgeball) ?l - curved_wooden_ramp) (in_motion ?c) ) )
                (once (< (distance ?p ?c) 0.5) )
              )
            )
          )
        )
        (preference preferenceD
          (exists (?m - pillow)
            (then
              (hold-while (and (not (agent_holds ?m) ) (not (agent_holds ?c agent) ) ) (not (agent_holds ?c ?c) ) )
              (hold (not (not (not (on tan) ) ) ) )
              (once (and (in agent ?m) (on ?c) (not (= (distance desk 9 room_center) 4) ) ) )
            )
          )
        )
        (preference preferenceE
          (exists (?w ?q ?t ?u ?i ?o - (either doggie_bed alarm_clock))
            (exists (?v - curved_wooden_ramp)
              (then
                (once (= (distance ?o ?c) 1 (distance ?t 10)) )
                (once (agent_holds ?t ?q) )
                (once (adjacent ?t) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preferenceA:green) (count-once-per-objects preferenceA:dodgeball) )
)
(:scoring
  (+ (count preferenceD:beachball) (count-once-per-objects preferenceE:red) )
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
                        (toggled_on ?xxx)
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
                      (exists (?k - (either pen doggie_bed doggie_bed dodgeball))
                        (on ?k ?k)
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
    (>= (count preferenceA:golfball:yellow:doggie_bed) (count-once preferenceA:yellow) )
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
    (forall (?m - hexagonal_bin)
      (game-conserved
        (in ?m ?m)
      )
    )
    (forall (?h ?d - hexagonal_bin)
      (exists (?t - hexagonal_bin)
        (and
          (forall (?w - ball)
            (game-optional
              (in_motion ?h)
            )
          )
          (exists (?i - chair)
            (exists (?m - building)
              (exists (?q - (either cd beachball book dodgeball))
                (game-conserved
                  (in ?d desk)
                )
              )
            )
          )
        )
      )
    )
    (forall (?q - (either cylindrical_block dodgeball))
      (and
        (game-conserved
          (on ?q)
        )
        (game-conserved
          (touch ?q ?q)
        )
        (and
          (and
            (game-conserved
              (is_setup_object ?q ?q)
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
      (exists (?n - game_object ?h - hexagonal_bin)
        (exists (?m - (either teddy_bear mug dodgeball))
          (exists (?w - hexagonal_bin ?y - (either dodgeball blue_cube_block pyramid_block yellow))
            (then
              (once (same_color ?m ?m) )
              (once (and (in ?h) (adjacent ?m ?h) (and (and (on ?h ?y) (and (and (and (same_color ?m) (agent_holds ?m) ) (in_motion ?m) ) (and (not (same_color green_golfball ?y) ) (exists (?b - (either triangular_ramp)) (on ?h) ) ) ) ) (and (in_motion ?m) (not (in ?m ?h) ) ) ) (and (or (and (agent_holds bed ?h) (in_motion ?y) ) (not (in_motion ?y ?h) ) ) (agent_holds ?h) ) ) )
              (once (in_motion rug ?h) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?d - hexagonal_bin)
        (exists (?r ?f ?g ?s - shelf ?g - dodgeball ?b - hexagonal_bin)
          (then
            (hold-while (not (on ?b) ) (in_motion ?b) )
            (once (and (agent_holds ?b) (exists (?l - (either mug hexagonal_bin) ?p - curved_wooden_ramp) (and (agent_holds ?d ?b) (not (and (in rug) (not (agent_crouches ?b ?b) ) (and (exists (?h - (either yellow_cube_block golfball)) (not (on desk bed) ) ) (adjacent agent) ) (adjacent ?d ?p) ) ) ) ) (agent_holds ?b ?d) (agent_holds ?d ?d) (and (not (on agent ?d) ) (not (in_motion ?b agent) ) ) (broken ?b ?d) (not (on ?d) ) ) )
            (once (in_motion ?d ?d) )
          )
        )
      )
    )
    (forall (?s - hexagonal_bin ?r - ball ?l - hexagonal_bin)
      (and
        (preference preferenceC
          (exists (?a - cube_block ?b - hexagonal_bin ?i - (either book pyramid_block) ?n - cylindrical_block)
            (then
              (hold (agent_holds ?n) )
              (forall-sequence (?e - cylindrical_block ?p - curved_wooden_ramp)
                (then
                  (hold-while (in ?l agent) (on ?l ?p) )
                  (once (and (agent_holds ?p) (not (agent_holds agent) ) ) )
                  (once (exists (?g - dodgeball) (on desk) ) )
                )
              )
              (hold-while (and (not (and (on ?l) (and (not (or (not (not (on ?l) ) ) (and (and (agent_holds ?l ?l) (and (or (= 0.5 1) (>= 1 4) ) (in ?n) (and (and (agent_holds ?n agent) (agent_holds ?n ?l) ) (not (in ?n) ) ) ) ) (same_color ?n) ) ) ) (not (on ?l) ) ) ) ) (in_motion ?n ?n) (and (in rug) (in_motion ?n) ) (in ?n ?n) (in_motion ?l) (and (and (not (> 1 2) ) (and (on ?l) (not (and (in_motion ?l) (agent_holds ?n) ) ) ) ) (same_type ?l ?l) ) ) (and (agent_holds ?n) (on ?n) ) )
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
  (count preferenceA:golfball)
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
    (>= (* (count preferenceA:pink) 10 (* (count-once-per-objects preferenceA:yellow) 0 )
      )
      (count preferenceA:doggie_bed)
    )
    (>= (count preferenceA:yellow) 4 )
    (or
      (>= 25 (count-same-positions preferenceA:blue_dodgeball) )
      (> 2 (+ 4 2 )
      )
    )
    (>= (* (count preferenceA:basketball:book) (count-once preferenceA:green) )
      (<= (count-once preferenceA:pink_dodgeball:golfball) (count preferenceA:beachball:pink) )
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
      (forall (?h - dodgeball ?b - game_object)
        (and
          (forall (?i - beachball ?m - hexagonal_bin)
            (forall (?x - block)
              (and
                (forall (?w - (either yellow pillow))
                  (exists (?h - hexagonal_bin)
                    (and
                      (game-optional
                        (and
                          (touch ?h)
                          (not
                            (and
                              (= (distance ?m ?h 10) (distance ))
                              (not
                                (not
                                  (agent_holds ?w)
                                )
                              )
                              (not
                                (adjacent_side ?w ?m)
                              )
                            )
                          )
                        )
                      )
                      (game-optional
                        (not
                          (and
                            (in_motion ?w)
                            (on ?b)
                          )
                        )
                      )
                      (exists (?a - pyramid_block)
                        (or
                          (and
                            (exists (?c - book ?n - dodgeball)
                              (game-conserved
                                (not
                                  (and
                                    (on ?a)
                                    (and
                                      (in_motion ?x ?n ?b)
                                      (and
                                        (< 1 4)
                                        (and
                                          (not
                                            (and
                                              (in ?m)
                                              (in ?m)
                                              (not
                                                (and
                                                  (agent_holds ?n)
                                                  (and
                                                    (and
                                                      (game_start ?b)
                                                      (not
                                                        (in_motion ?n ?m)
                                                      )
                                                    )
                                                    (= 1 (distance 7 7))
                                                  )
                                                  (adjacent ?h)
                                                )
                                              )
                                            )
                                          )
                                          (not
                                            (in_motion ?x rug)
                                          )
                                        )
                                      )
                                    )
                                    (and
                                      (in_motion ?x)
                                      (not
                                        (< 1 (distance agent ?b))
                                      )
                                    )
                                  )
                                )
                              )
                            )
                          )
                          (or
                            (game-conserved
                              (in_motion ?a)
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
            (exists (?w - red_dodgeball)
              (forall (?s - hexagonal_bin)
                (exists (?g - block)
                  (and
                    (exists (?m ?k - wall)
                      (and
                        (game-conserved
                          (and
                            (agent_holds ?g)
                            (= (distance_side ?w agent))
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
            (and
              (exists (?f - dodgeball)
                (and
                  (forall (?m - hexagonal_bin)
                    (game-conserved
                      (in_motion ?m ?b)
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
                  (forall (?s - red_dodgeball)
                    (adjacent ?s ?b)
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
          (exists (?y ?h - (either dodgeball hexagonal_bin rug))
            (exists (?v - dodgeball)
              (and
                (exists (?z - dodgeball ?u - hexagonal_bin)
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
  (>= (* (count preferenceA:basketball:basketball) )
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
    (exists (?v ?u - (either laptop golfball))
      (game-conserved
        (or
          (not
            (and
              (not
                (< 10 6)
              )
              (exists (?l ?w - game_object)
                (agent_holds ?l ?s)
              )
            )
          )
          (agent_holds ?v)
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
        (hold-while (same_object ?xxx ?xxx) (in_motion ?xxx ?xxx) (and (and (object_orientation ?xxx ?xxx) (not (not (not (not (and (and (in_motion ?xxx ?xxx) (and (and (agent_holds ?xxx ?xxx) (or (not (on sideways ?xxx) ) (in ?xxx) ) ) (= 9 2 7) ) ) (not (not (agent_holds ?xxx) ) ) ) ) ) ) ) ) (in_motion agent) (same_color ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (+ (+ (or 10 (+ 4 )
          (+ (* (count-once-per-objects preferenceA:doggie_bed) 8 )
            (* 2 (count preferenceA:basketball) (count-once-per-objects preferenceA:dodgeball:book) )
          )
        )
        50
        4
        (* (total-score) 3 )
        (count preferenceA:tall_cylindrical_block)
        (= (+ (- (+ 7 (count preferenceA:yellow) )
            )
            (count-once preferenceA:golfball)
          )
          5
        )
      )
      (count preferenceA:golfball:hexagonal_bin)
    )
    (* (+ (* 9 (count preferenceA:red) )
        (count preferenceA:golfball)
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
      (exists (?j - hexagonal_bin)
        (game-optional
          (and
            (touch ?j)
            (not
              (agent_holds ?j)
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
          (forall (?k - dodgeball)
            (forall (?e - (either doggie_bed doggie_bed) ?x - hexagonal_bin)
              (forall (?q - wall)
                (exists (?d - ball)
                  (exists (?y - hexagonal_bin ?f - hexagonal_bin ?i - chair)
                    (game-optional
                      (in ?i ?k)
                    )
                  )
                )
              )
            )
          )
          (exists (?b ?s - block ?q - hexagonal_bin)
            (and
              (exists (?h - cube_block)
                (game-optional
                  (touch floor)
                )
              )
            )
          )
          (forall (?t - (either blue_cube_block pyramid_block) ?a - hexagonal_bin)
            (not
              (forall (?k - (either tall_cylindrical_block laptop) ?x - flat_block)
                (and
                  (and
                    (exists (?f - shelf)
                      (game-conserved
                        (in ?f rug floor)
                      )
                    )
                    (game-conserved
                      (in_motion ?a)
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
        (exists (?x ?m - cube_block)
          (exists (?z - drawer)
            (and
              (game-conserved
                (in ?z ?m)
              )
              (and
                (and
                  (forall (?b - hexagonal_bin)
                    (and
                      (forall (?u - (either bed cd) ?i - ball ?v ?e - pyramid_block)
                        (forall (?t - (either cube_block dodgeball hexagonal_bin watch))
                          (exists (?f - hexagonal_bin)
                            (game-optional
                              (adjacent ?f ?f)
                            )
                          )
                        )
                      )
                    )
                  )
                  (exists (?k - hexagonal_bin)
                    (and
                      (and
                        (exists (?t - triangular_ramp ?h ?g - wall ?t - hexagonal_bin)
                          (exists (?d - hexagonal_bin)
                            (game-conserved
                              (in_motion ?z ?x)
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
              (exists (?j - game_object)
                (and
                  (and
                    (game-optional
                      (and
                        (not
                          (and
                            (in_motion bed)
                            (agent_holds ?x)
                          )
                        )
                        (and
                          (not
                            (not
                              (in ?j bed)
                            )
                          )
                          (in_motion bed ?j)
                        )
                      )
                    )
                    (and
                      (game-optional
                        (and
                          (on ?z)
                          (not
                            (not
                              (not
                                (agent_holds ?x)
                              )
                            )
                          )
                        )
                      )
                      (exists (?u - chair)
                        (game-optional
                          (not
                            (adjacent ?j)
                          )
                        )
                      )
                    )
                  )
                  (not
                    (game-optional
                      (not
                        (touch ?j ?x)
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
      (exists (?m ?k - bridge_block)
        (then
          (once (in_motion agent ?m) )
          (once (and (not (not (in ?m) ) ) (agent_holds ?k ?m) ) )
          (once (forall (?b - (either pyramid_block red)) (in_motion ?k ?b) ) )
        )
      )
    )
    (forall (?a - cube_block ?s - color ?o - red_pyramid_block)
      (and
        (preference preferenceB
          (exists (?n - dodgeball ?x - (either cd dodgeball golfball laptop) ?f - game_object ?j - doggie_bed)
            (then
              (once (in agent ?o) )
              (once (and (in ?j) (in_motion ?j) (exists (?p - cube_block) (on desk) ) ) )
              (once (in_motion ?o ?o) )
            )
          )
        )
        (preference preferenceC
          (then
            (hold (or (and (rug_color_under ?o ?o) (in ?o) (and (and (agent_holds ?o) ) (in ?o) ) (and (not (in_motion ?o) ) (in_motion ?o ?o) ) ) (object_orientation ?o ?o) (same_color ?o front) ) )
            (once (in_motion ?o ?o) )
            (once (not (and (agent_holds upright) (on ?o ?o) ) ) )
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
  (>= 1 (count preferenceA:orange:basketball) )
)
(:scoring
  (- 5 )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (game_start ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preferenceA
      (then
        (once-measure (agent_holds ?xxx) (distance ?xxx room_center) )
        (once (game_over ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count preferenceA:yellow:pink) (external-forall-minimize (count-once-per-objects preferenceA:yellow_pyramid_block) ) )
)
(:scoring
  (count preferenceA:beachball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (game-conserved
      (on ?xxx)
    )
    (exists (?w - building ?h - (either pyramid_block book))
      (game-optional
        (not
          (agent_holds ?h)
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
    (forall (?r - ball ?e - tall_cylindrical_block)
      (and
        (preference preferenceA
          (then
            (once (not (agent_holds ?e) ) )
            (hold (touch ?e ?e) )
            (once (in ?e ?e) )
            (once (and (exists (?n ?i - hexagonal_bin) (in_motion bed desk) ) (open ?e) ) )
          )
        )
      )
    )
    (forall (?b - dodgeball)
      (and
        (preference preferenceB
          (then
            (hold (exists (?w - red_pyramid_block) (not (in_motion ?b) ) ) )
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
  (>= (count-total preferenceB:orange) (count preferenceA:basketball) )
)
(:scoring
  (count preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-optional
    (exists (?l - hexagonal_bin)
      (on ?l ?l)
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
  (> (count preferenceA:green) 1 )
)
(:scoring
  3
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?s - block ?l - building)
    (game-optional
      (and
        (not
          (< 1 (distance 5 room_center))
        )
        (in_motion agent ?l)
      )
    )
  )
)
(:constraints
  (and
    (forall (?p - dodgeball)
      (and
        (preference preferenceA
          (exists (?g ?l ?u ?x ?d ?o - chair)
            (at-end
              (agent_holds ?g)
            )
          )
        )
        (preference preferenceB
          (then
            (once (agent_holds ?p agent) )
            (once (in_motion ?p ?p) )
            (once (in_motion ?p) )
          )
        )
        (preference preferenceC
          (exists (?m - block)
            (at-end
              (same_type ?m ?m)
            )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?j - (either dodgeball dodgeball))
        (then
          (once (agent_holds ?j ?j) )
          (once (agent_holds ?j) )
          (any)
        )
      )
    )
    (preference preferenceE
      (then
        (hold (not (agent_holds ?xxx) ) )
        (once (game_start ?xxx ?xxx) )
        (once (object_orientation ?xxx) )
      )
    )
    (forall (?v - hexagonal_bin)
      (and
        (preference preferenceF
          (exists (?g - cube_block)
            (exists (?n ?r ?q - cube_block ?l ?t - (either cylindrical_block hexagonal_bin laptop))
              (then
                (once (in_motion ?t) )
                (once (in_motion ?l) )
                (once (on ?t ?g) )
              )
            )
          )
        )
      )
    )
    (preference preferenceG
      (exists (?c - hexagonal_bin)
        (then
          (once (not (not (not (on agent) ) ) ) )
          (once (or (not (and (in_motion ?c) (in_motion ?c ?c) (exists (?m - curved_wooden_ramp) (or (adjacent ?c) (in_motion bed ?m) ) ) ) ) (exists (?f - game_object) (in_motion ?c ?f ?c) ) ) )
          (once (forall (?n - chair) (on ?n) ) )
          (once (in ?c ?c) )
        )
      )
    )
    (forall (?p - dodgeball ?s - beachball)
      (and
        (preference preferenceH
          (then
            (hold (or (game_over agent ?s) (in ?s) ) )
            (hold (agent_holds desk) )
            (hold (= (building_size ) 0.5 (distance ?s agent)) )
          )
        )
        (preference preferenceI
          (exists (?q - ball)
            (then
              (once (in_motion ?q) )
              (hold (in_motion pink ?s) )
              (once (and (agent_holds ?s ?q) (not (and (> 1 3) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count preferenceA:doggie_bed) 5 )
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
      (exists (?v - hexagonal_bin)
        (then
          (hold (not (and (and (and (forall (?o - hexagonal_bin) (and (not (exists (?c - hexagonal_bin) (agent_holds ?c ?c) ) ) (in ?v ?o) (exists (?p - doggie_bed ?b - game_object) (not (object_orientation agent) ) ) (touch ?o) (agent_holds ?v ?v) (and (in ?o) (in upright floor ?o ?o) ) ) ) (faces ?v) ) (not (same_object agent) ) ) (agent_holds ?v ?v) ) ) )
          (once (= 2 1) )
          (hold (agent_holds ?v ?v) )
        )
      )
    )
  )
)
(:terminal
  (> (- 1 )
    (count preferenceA:dodgeball)
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
    (forall (?u - block)
      (and
        (preference preferenceA
          (then
            (once (agent_holds rug) )
            (once (not (agent_holds ?u) ) )
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
    (exists (?q - block)
      (exists (?f - dodgeball ?p - hexagonal_bin ?k - ball)
        (game-optional
          (not
            (same_type ?k)
          )
        )
      )
    )
    (exists (?o - (either teddy_bear pen))
      (game-conserved
        (and
          (not
            (touch ?o ?o)
          )
          (same_color ?o)
        )
      )
    )
    (game-conserved
      (exists (?m - hexagonal_bin)
        (not
          (agent_holds ?m ?m)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?x - color)
        (exists (?j - block ?k - dodgeball)
          (exists (?t - dodgeball)
            (exists (?y - cube_block)
              (then
                (once (and (in_motion ?x) (not (adjacent ?t) ) (faces ?y) ) )
                (once (agent_holds rug ?y) )
                (once (in_motion ?y) )
              )
            )
          )
        )
      )
    )
    (forall (?q ?y - (either key_chain golfball tall_cylindrical_block teddy_bear curved_wooden_ramp cube_block main_light_switch))
      (and
        (preference preferenceB
          (exists (?e - dodgeball)
            (then
              (hold (in_motion ?e agent) )
              (once (agent_holds rug) )
              (hold-while (on ?y ?q) (in_motion ?y) )
            )
          )
        )
      )
    )
    (forall (?a - red_dodgeball ?m - wall ?u - hexagonal_bin)
      (and
        (preference preferenceC
          (exists (?z - game_object)
            (then
              (once (and (agent_holds door floor) (in_motion ?u ?z) ) )
              (once (in_motion blue) )
              (once (same_color ?z ?u pillow ?z) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (* (count preferenceB:pink:green:orange) (count-once-per-objects preferenceC:yellow_cube_block) )
      50
    )
    (>= 2 (total-time) )
    (< 30 (count preferenceC:dodgeball) )
  )
)
(:scoring
  (* (* (- 2 )
    )
    (* (* (count-once preferenceA:golfball) (count-once-per-objects preferenceA:blue_dodgeball:pink) )
      (count-unique-positions preferenceB:dodgeball)
      10
      (count preferenceB:green:golfball:red_pyramid_block)
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
  (exists (?s - color)
    (game-conserved
      (in_motion ?s top_shelf)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?k - shelf)
        (exists (?b - (either dodgeball triangle_block pillow) ?v - triangular_ramp)
          (exists (?w - building)
            (then
              (once (adjacent ?w) )
              (once (not (and (not (not (adjacent ?w) ) ) (adjacent pillow) ) ) )
              (hold (not (and (= (distance ?w) (distance ?v ?w)) (agent_holds ?w) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (external-forall-maximize 4 ) (count preferenceA:pink:dodgeball) )
      6
    )
  )
)
(:scoring
  (* (count preferenceA:golfball:doggie_bed) )
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
      (exists (?y - cube_block)
        (then
          (hold (adjacent ?y top_drawer ?y) )
          (hold (in_motion ?y ?y) )
          (once (in_motion ?y) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (and (on door) (agent_holds ?xxx) ) )
        (once (in ?xxx ?xxx) )
        (once (exists (?k - (either pyramid_block alarm_clock)) (faces ?k) ) )
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
      (exists (?p - hexagonal_bin)
        (then
          (once (not (on ?p) ) )
          (once (in_motion desktop ?p ?p) )
          (once (agent_holds ?p ?p) )
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
          (exists (?k - cube_block)
            (exists (?x - golfball ?t - dodgeball)
              (exists (?x - (either pyramid_block pencil))
                (then
                  (once (or (in bed) (in_motion ?t) ) )
                  (hold (same_color ?t) )
                  (once (and (not (not (adjacent rug) ) ) (and (and (in ?b) (not (and (exists (?f - dodgeball) (in_motion ?k ?f) ) (in_motion ?x) ) ) (in_motion ?x west_wall) ) (adjacent ?x ?t) (on ?b) ) (not (in_motion ?t ?t) ) (not (touch ?x ?x) ) (in ?b ?b) (not (in_motion desk) ) (agent_holds door ?t) (agent_holds ?t) ) )
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
  (>= (+ (count-once-per-objects preferenceA:purple) (* (count-once-per-objects preferenceC:cylindrical_block) 12 3 (count-overlapping preferenceA:basketball) (* (count preferenceA:golfball) (* (not 10 ) (count-once-per-objects preferenceC:golfball) )
        )
        (or
          (count preferenceA:pink_dodgeball)
          (count preferenceC:yellow_cube_block:beachball)
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
  (forall (?p - game_object)
    (forall (?l - (either alarm_clock cylindrical_block))
      (game-optional
        (and
          (in_motion ?p)
          (agent_holds ?l)
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
    (forall (?g - hexagonal_bin ?b - ball)
      (and
        (preference preferenceB
          (exists (?r - dodgeball ?f - hexagonal_bin)
            (then
              (once (not (not (and (agent_holds pink desk) (in ?b ?f) ) ) ) )
              (once (and (in_motion ?f ?b) (agent_holds ?f) ) )
              (once (not (same_type ?f) ) )
              (once (not (in_motion agent ?b) ) )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?z - pillow)
        (then
          (once (adjacent_side ?z) )
          (once (object_orientation ?z) )
        )
      )
    )
  )
)
(:terminal
  (or
    (= (+ (count-once-per-objects preferenceA:purple) (count preferenceC:cube_block:golfball) )
      3
    )
    (>= (and 2 15 8 ) 1 )
    (>= 18 (count preferenceC:dodgeball) )
  )
)
(:scoring
  (count-once-per-external-objects preferenceB:basketball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?f - ball)
    (forall (?u ?h - hexagonal_bin)
      (game-conserved
        (forall (?w - game_object)
          (in_motion ?f ?f)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?g - ball)
        (exists (?x - ball)
          (then
            (hold (in_motion ?g) )
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
      (exists (?w - doggie_bed)
        (at-end
          (and
            (object_orientation back ?w)
            (on ?w)
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
      (exists (?z - building)
        (at-end
          (not
            (in_motion ?z)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preferenceA:beachball) (count preferenceA:pink) )
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
    (forall (?w - block)
      (and
        (preference preferenceA
          (then
            (once (< 6 1) )
            (once (and (not (agent_holds ?w ?w) ) (agent_holds ?w ?w) ) )
            (once (exists (?m - building) (on ?w ?m) ) )
          )
        )
        (preference preferenceB
          (then
            (once-measure (in_motion ?w ?w) (distance 2 door) )
            (once (not (same_type rug) ) )
            (once (same_color ?w ?w) )
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
  (* (count-once preferenceB:beachball) (* (count preferenceB:orange:beachball) (< 7 6 )
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?m ?o ?f - teddy_bear ?z - wall)
    (exists (?n - (either pyramid_block alarm_clock))
      (exists (?m - dodgeball ?l - hexagonal_bin)
        (game-optional
          (and
            (and
              (same_color ?n)
              (< 1 (distance ))
              (exists (?r - game_object)
                (< 1 1)
              )
              (in_motion ?l agent)
            )
            (on ?z right)
            (not
              (agent_holds ?l)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?y - block ?t - drawer ?h - book ?z - hexagonal_bin ?w - ball)
      (and
        (preference preferenceA
          (exists (?k - drawer ?f ?q ?j - triangular_ramp)
            (exists (?t - dodgeball)
              (then
                (hold (and (agent_holds rug) (agent_holds ?t) ) )
                (once (< 2 4) )
                (once (on ?f ?t) )
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
      (>= 3 (+ 9 (* (count preferenceA:yellow:dodgeball) (+ 10 (count preferenceA:dodgeball) (* 3 (count-once-per-objects preferenceA:blue_cube_block) )
              10
              9
              8
              (* (+ (* (count preferenceA:golfball) (count-once-per-objects preferenceA:blue_pyramid_block) )
                  (count-once preferenceA:basketball:dodgeball)
                )
                180
              )
              (count preferenceA:dodgeball)
              (count preferenceA:basketball)
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
      (exists (?z - ball)
        (then
          (hold-while (and (in ?z) (and (and (and (< (distance ?z) (distance ?z ?z)) (on ?z ?z) ) (in ?z ?z) ) (and (in_motion left ?z) (not (between ?z) ) ) (in agent) ) (and (on ?z) (and (same_type ?z ?z) (not (and (and (in desk) (on ?z) (agent_holds ?z) ) (on pillow) (not (on ?z ?z) ) (and (and (exists (?j - dodgeball) (adjacent ?z ?j) ) (agent_holds ?z ?z) ) (and (> 2 (distance room_center door)) (in ?z) ) (and (and (rug_color_under ?z) (adjacent ?z) ) (> (distance ?z 9) 2) ) ) (in_motion pink) (in ?z) (and (adjacent ?z) (and (touch ?z ?z) ) ) ) ) ) ) (adjacent_side ?z) ) (not (same_color ?z) ) )
          (once (same_object ?z ?z agent) )
          (once (not (in ?z ?z) ) )
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
    (> (* (count preferenceC:dodgeball) (external-forall-maximize (count preferenceC:beachball) ) )
      (count-once-per-objects preferenceC:beachball:pink_dodgeball)
    )
  )
)
(:scoring
  (count preferenceB)
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
    (forall (?k - dodgeball)
      (and
        (preference preferenceA
          (then
            (once (faces ?k ?k) )
            (once (agent_holds ?k ?k ?k) )
            (once (in_motion ?k) )
          )
        )
        (preference preferenceB
          (then
            (once (and (touch ?k) (agent_holds pink) ) )
            (hold-while (forall (?s - (either cellphone desktop mug dodgeball yellow desktop dodgeball)) (adjacent ?s) ) (in_motion ?k) (on ?k) )
            (once (and (in ?k) (and (on desk floor) (in ?k) ) ) )
          )
        )
      )
    )
    (forall (?o - golfball)
      (and
        (preference preferenceC
          (exists (?a - game_object)
            (exists (?t - doggie_bed)
              (exists (?h - drawer)
                (then
                  (hold (in_motion agent ?a) )
                  (once (and (agent_holds ?a) (on ?o) ) )
                  (hold (agent_holds ?o ?o) )
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
  (>= (* 7 5 (count preferenceB:blue_dodgeball) (count-once-per-objects preferenceC:dodgeball:basketball) 3 (+ (+ (count preferenceA:yellow) (= 5 4 )
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
  (> 30 (count preferenceB:cube_block:basketball:golfball) )
)
(:scoring
  (count preferenceA:basketball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?h - drawer)
    (game-conserved
      (and
        (in_motion ?h ?h)
        (in_motion green_golfball)
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?u ?o - (either cd golfball blue_cube_block yellow_cube_block))
        (exists (?d - hexagonal_bin)
          (then
            (once (not (and (in desk) (and (toggled_on ?d) (not (touch ?o ?d) ) ) ) ) )
            (hold-while (agent_holds ?o) (or (not (in_motion ?d ?u) ) ) (and (exists (?i - block) (not (and (agent_holds ?o ?i) (in_motion ?u ?o) ) ) ) (broken ?u) (in ?u) ) )
            (once (not (not (and (in ?d yellow) (agent_holds ?d) ) ) ) )
          )
        )
      )
    )
    (forall (?r - teddy_bear ?f - game_object)
      (and
        (preference preferenceB
          (then
            (once (not (adjacent ) ) )
            (hold (in_motion ?f ?f) )
            (hold (not (in_motion floor ?f) ) )
          )
        )
        (preference preferenceC
          (at-end
            (on ?f ?f)
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (and
      (>= (= 3 (count preferenceC:pyramid_block) 5 )
        (external-forall-minimize
          (* 4 18 )
        )
      )
      (>= (count-once-per-objects preferenceA:pink:rug) (count preferenceB) )
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
      (forall (?l - hexagonal_bin)
        (forall (?e - tall_cylindrical_block)
          (exists (?z - hexagonal_bin)
            (game-conserved
              (and
                (adjacent ?z ?b)
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
    (forall (?s - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?k - (either blue_cube_block wall))
            (exists (?w - shelf)
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
  (>= 8 (- (count preferenceB:cube_block) )
  )
)
(:scoring
  (count-once-per-objects preferenceB:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?y - (either tall_cylindrical_block curved_wooden_ramp) ?k - shelf ?b - teddy_bear)
    (forall (?a - curved_wooden_ramp)
      (game-conserved
        (agent_holds ?b)
      )
    )
  )
)
(:constraints
  (and
    (forall (?u - cube_block)
      (and
        (preference preferenceA
          (exists (?e - blue_pyramid_block ?p - tall_cylindrical_block ?o - building ?j - teddy_bear)
            (exists (?f - dodgeball)
              (then
                (hold-while (and (in_motion ?j ?f) (exists (?i - hexagonal_bin ?t - dodgeball) (and (agent_holds ?t ?f) (and (not (in_motion desktop) ) (not (not (and (in_motion ?t) (touch ?t) ) ) ) ) ) ) ) (and (not (on ?f) ) (not (< 5 1) ) (agent_holds ?j ?f) ) )
                (once (not (agent_holds ?u rug) ) )
                (hold (and (< (distance ?f ?j) (distance ?u desk)) (not (and (same_type side_table ?f) (exists (?p - hexagonal_bin) (adjacent ?u) ) ) ) ) )
              )
            )
          )
        )
      )
    )
    (forall (?d - hexagonal_bin)
      (and
        (preference preferenceB
          (exists (?x - rug)
            (exists (?j - hexagonal_bin ?n - (either tall_cylindrical_block triangle_block))
              (exists (?w ?q - hexagonal_bin)
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
                (adjacent_side ?xxx ?xxx)
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
              (faces ?xxx)
              (exists (?g - hexagonal_bin)
                (and
                  (and
                    (not
                      (adjacent ?g)
                    )
                    (not
                      (in_motion ?g ?g)
                    )
                    (>= 9 1)
                    (on ?g)
                  )
                  (and
                    (in_motion ?g ?g)
                    (in ?g ?g)
                    (not
                      (between ?g ?g yellow)
                    )
                  )
                )
              )
            )
          )
        )
        (exists (?r ?u - book)
          (game-optional
            (exists (?m - cube_block ?h - red_pyramid_block)
              (and
                (not
                  (in_motion left ?h)
                )
                (agent_holds ?r)
                (and
                  (agent_holds ?u ?h)
                  (or
                    (agent_holds bed)
                    (and
                      (in ?u)
                      (on ?r)
                    )
                    (agent_holds ?u)
                  )
                )
              )
            )
          )
        )
        (forall (?p - (either pencil))
          (exists (?q - hexagonal_bin ?u - cube_block)
            (game-optional
              (in_motion ?p agent)
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
                  (forall (?m - hexagonal_bin)
                    (and
                      (game-optional
                        (and
                          (agent_holds bed ?m)
                          (and
                            (agent_holds ?b)
                            (not
                              (not
                                (not
                                  (and
                                    (forall (?h - ball)
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
                            (on ?b ?m ?m)
                            (in_motion desk)
                            (on ?m blue)
                            (exists (?x - curved_wooden_ramp)
                              (and
                                (and
                                  (agent_holds ?m)
                                  (on ?x ?b)
                                )
                              )
                            )
                            (and
                              (exists (?e - hexagonal_bin)
                                (not
                                  (in_motion ?e ?m)
                                )
                              )
                              (agent_holds pink_dodgeball)
                              (> (distance room_center ?b) (distance room_center 2))
                            )
                            (and
                              (in bed)
                              (in upside_down ?m floor)
                            )
                            (in desk)
                            (agent_holds ?b ?m)
                          )
                          (in ?m)
                        )
                      )
                      (game-conserved
                        (= (distance ?m ?b) 1)
                      )
                      (exists (?q - game_object)
                        (and
                          (exists (?l - golfball)
                            (game-optional
                              (agent_holds ?l)
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
              (forall (?w - hexagonal_bin)
                (and
                  (forall (?v - hexagonal_bin)
                    (and
                      (exists (?l - (either beachball desktop))
                        (and
                          (forall (?n - hexagonal_bin)
                            (and
                              (game-optional
                                (and
                                  (and
                                    (and
                                      (agent_holds ?w)
                                      (and
                                        (on floor)
                                        (on ?l ?n)
                                      )
                                    )
                                    (agent_holds ?n ?l)
                                  )
                                  (agent_holds ?b)
                                )
                              )
                              (exists (?d - hexagonal_bin)
                                (game-conserved
                                  (and
                                    (and
                                      (in_motion ?l ?n)
                                      (in ?n top_shelf)
                                    )
                                    (adjacent ?n)
                                  )
                                )
                              )
                              (game-optional
                                (in_motion ?w ?v)
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
                    (forall (?c - doggie_bed)
                      (exists (?i - hexagonal_bin ?w - chair)
                        (game-conserved
                          (in_motion ?w)
                        )
                      )
                    )
                  )
                )
                (forall (?h - (either chair))
                  (and
                    (game-optional
                      (in_motion ?h ?h ?b)
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
      (exists (?j - wall ?y - hexagonal_bin ?y - hexagonal_bin ?z - dodgeball)
        (exists (?k - dodgeball)
          (exists (?u - hexagonal_bin)
            (then
              (once (in_motion agent ?u) )
              (hold (in_motion pink) )
              (once (exists (?n - (either credit_card key_chain)) (and (not (not (on ?z) ) ) (and (same_color ?n agent) (same_color ?k) ) ) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?t - hexagonal_bin)
        (exists (?y - hexagonal_bin ?u ?c - game_object)
          (then
            (once (and (exists (?i - hexagonal_bin) (agent_holds floor sideways) ) (not (and (< (distance ?u 8) 1) (and (in_motion ?c ?t ?t) (on ?t) (same_type ?u ?c) (agent_holds agent ?c) ) (on ?t) (is_setup_object agent) ) ) ) )
            (once (not (or (and (and (agent_holds ?u rug) (agent_holds ?u ?c) ) (or (agent_holds ?u) (< (distance 6) (distance ?u ?t)) ) (not (in ?t ?u) ) ) (or (< 0.5 0.5) (in ?c) ) ) ) )
            (once (on ?c) )
          )
        )
      )
    )
  )
)
(:terminal
  (= 30 (count preferenceA:basketball) )
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
      (exists (?z - dodgeball)
        (then
          (once (not (and (not (in_motion ?z) ) (not (agent_holds ?z) ) ) ) )
          (hold (in_motion ?z) )
          (once (and (is_setup_object ?z ?z) (exists (?i - block) (in_motion ?z ?i) ) (and (agent_holds ?z) (agent_holds ?z) ) ) )
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
      (exists (?x ?m - building ?f - doggie_bed)
        (exists (?n - hexagonal_bin ?y - (either hexagonal_bin cellphone))
          (exists (?k - building)
            (then
              (hold-while (in_motion ?k ?f) (on rug ?f) )
              (once (not (in_motion ?f) ) )
              (hold (agent_holds ?k ?k) )
            )
          )
        )
      )
    )
    (preference preferenceE
      (then
        (once (agent_holds ?xxx) )
        (hold-while (and (exists (?z - chair) (on ?z) ) (adjacent_side ?xxx ?xxx) ) (in bed ?xxx) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
    (preference preferenceF
      (exists (?y - (either cd alarm_clock) ?s - hexagonal_bin ?g - curved_wooden_ramp ?h - hexagonal_bin)
        (exists (?o - hexagonal_bin ?e - pyramid_block)
          (then
            (once (in_motion ?e) )
            (once (and (is_setup_object ?e front) (and (in_motion ?h ?h) (not (and (agent_holds ?e ?h) (in desk) (< 1 (x_position ?e 3)) (or (in_motion ?h) (and (in_motion ?e ?e ?e) (and (exists (?j - dodgeball ?q ?r ?y - dodgeball ?s ?v - curved_wooden_ramp ?b - building) (<= 2 5) ) (< 2 (distance ?h agent)) ) ) ) (on ?h) (not (not (in ?e) ) ) (in_motion ?h) (exists (?l - (either hexagonal_bin bridge_block)) (on ?e) ) ) ) ) ) )
            (hold (in_motion ?e) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-same-positions preferenceC:yellow) (count-shortest preferenceE:triangle_block:dodgeball) )
    (>= (+ (count preferenceC:side_table) (/ (* (count-once-per-objects preferenceF:cube_block:pink) (count preferenceD:orange) )
          10
        )
        (* (* (* 1 (+ 3 (count-measure preferenceD:dodgeball:pink) (count preferenceE:golfball:dodgeball) (* (* (count preferenceA:beachball) (count-once-per-objects preferenceF:cube_block) )
                  40
                )
              )
            )
            15
          )
          (external-forall-maximize
            (not
              (count preferenceB:pink)
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
  (exists (?w - teddy_bear ?o - chair)
    (or
      (game-optional
        (or
          (in_motion ?o ?o)
          (in ?o ?o)
          (agent_holds ?o ?o)
        )
      )
      (exists (?y - hexagonal_bin)
        (and
          (exists (?u - hexagonal_bin ?g - hexagonal_bin)
            (game-optional
              (same_type ?y ?y)
            )
          )
        )
      )
      (and
        (game-conserved
          (in ?o)
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
          (hold (and (and (exists (?z - cube_block ?l - cube_block) (touch ?l ?b) ) (agent_holds ?b) ) (agent_holds ?b) ) )
        )
      )
    )
    (preference preferenceB
      (then
        (once (in_motion ?xxx) )
        (hold-while (and (exists (?m - teddy_bear ?m ?j - bridge_block) (and (agent_holds ?m ?m) (in agent) (equal_x_position ?j) (and (on ?j) (object_orientation bridge_block ?j) ) (not (in_motion ?m) ) (and (and (on ?m) (agent_holds ?j) ) (same_color ?m) (agent_holds ?j) ) (not (same_color ?j ?m) ) (agent_holds agent green) (on ?j) (and (on agent ?m) (in_motion agent ?m) ) ) ) (on ?xxx ?xxx desk ?xxx) ) (and (agent_holds desk) (= (distance 7 ?xxx) (distance bed ?xxx)) ) (in_motion ?xxx) )
        (hold (not (not (not (in_motion ?xxx ?xxx) ) ) ) )
      )
    )
  )
)
(:terminal
  (>= 10 (count preferenceB:golfball) )
)
(:scoring
  (* (* (- (count preferenceA:pink_dodgeball) )
      (count-measure preferenceB:basketball)
      (total-time)
      (or
        (count-once-per-objects preferenceA:cube_block)
      )
      7
      (+ 1 (count preferenceA:yellow) )
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
                (exists (?f - hexagonal_bin)
                  (exists (?w - curved_wooden_ramp)
                    (in_motion rug ?w)
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
      (= 6 (* (external-forall-maximize (count preferenceA:hexagonal_bin) ) (/ (count preferenceA:blue_dodgeball:yellow_cube_block) (* (count-once-per-objects preferenceA:red) (count-once preferenceA:golfball) )
          )
        )
      )
    )
    (>= (+ 30 (count preferenceA:purple) )
      3
    )
  )
)
(:scoring
  (count preferenceA:golfball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?t - dodgeball)
    (and
      (game-conserved
        (agent_holds ?t)
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
          (exists (?p - blinds ?z ?e - yellow_cube_block)
            (then
              (once (and (and (agent_holds ?z ?z) (forall (?b ?j - hexagonal_bin) (in_motion ?b ?j) ) (touch ?e top_shelf) ) (between ?z) ) )
              (once (in_motion ?z) )
              (any)
            )
          )
        )
        (preference preferenceC
          (exists (?b - golfball ?u - cube_block)
            (exists (?x ?b - doggie_bed)
              (then
                (once (in desk ?x) )
                (hold (on ?u) )
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
        (same_type ?e)
        (in_motion ?e)
        (exists (?j - dodgeball)
          (adjacent ?j ?e ?e)
        )
        (not
          (and
            (agent_holds ?e)
            (not
              (not
                (forall (?a - teddy_bear)
                  (not
                    (and
                      (not
                        (and
                          (in_motion right)
                          (and
                            (not
                              (agent_holds ?a)
                            )
                            (in ?a)
                          )
                          (not
                            (and
                              (not
                                (and
                                  (in_motion ?a)
                                  (not
                                    (in ?e ?e)
                                  )
                                )
                              )
                              (touch ?a agent)
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
      (exists (?c - hexagonal_bin)
        (exists (?r - ball ?a - curved_wooden_ramp)
          (exists (?u - dodgeball ?b - building)
            (exists (?j - teddy_bear)
              (exists (?y - dodgeball)
                (at-end
                  (and
                    (and
                      (adjacent bed)
                      (exists (?l - ball)
                        (in_motion ?c)
                      )
                    )
                    (and
                      (< (distance room_center ?y) 1)
                      (and
                        (and
                          (agent_holds agent ?c)
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
      (exists (?d - cylindrical_block)
        (then
          (once (< 1 (distance 7 ?d)) )
          (once (agent_holds south_wall ?d) )
          (once (not (above ?d) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preferenceA:hexagonal_bin) (* (+ (count preferenceA) 8 )
        7
      )
    )
    (count preferenceA:purple:book)
  )
)
(:scoring
  (+ (count-once-per-objects preferenceB:basketball) (+ (count-shortest preferenceA:pink) 5 (* 3 )
      (>= (count preferenceA:dodgeball:dodgeball) (count-once-per-objects preferenceA:beachball:pink_dodgeball) )
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
        (hold (and (equal_z_position ?xxx) (< 1 1) ) )
        (hold-while (and (in_motion ?xxx) (agent_holds bed agent) ) (agent_holds ?xxx) (in_motion ?xxx) )
      )
    )
    (preference preferenceB
      (then
        (once (agent_holds ?xxx) )
        (once (exists (?q - cube_block ?a - cube_block ?s - ball) (in_motion ?s ?s) ) )
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
  (forall (?k - (either lamp dodgeball key_chain) ?p - (either book red))
    (game-conserved
      (in_motion ?p desk ?p)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b ?o - ball ?g - (either dodgeball cube_block) ?s - red_pyramid_block)
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
      (count preferenceA:basketball)
    )
    (> (* 3 (total-score) )
      (external-forall-maximize
        3
      )
    )
  )
)
(:scoring
  (count preferenceA:basketball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?c ?w ?f - pyramid_block ?i - hexagonal_bin)
    (game-conserved
      (exists (?v - block ?m - doggie_bed ?j - hexagonal_bin)
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
            (once (and (not (game_over ?b ?b) ) (on ?b ?b) ) )
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
    (exists (?c - game_object)
      (and
        (game-optional
          (in_motion ?c ?c)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?t - doggie_bed)
        (at-end
          (on ?t)
        )
      )
    )
    (preference preferenceB
      (exists (?d - dodgeball ?q - hexagonal_bin)
        (then
          (hold-while (not (and (not (and (not (agent_holds ?q ?q) ) (in ?q) ) ) (touch ?q) ) ) (agent_holds ?q ?q) )
          (once (not (agent_holds ?q) ) )
          (hold (and (in_motion agent) (not (and (and (agent_holds ?q) (in_motion ?q) ) (toggled_on ) ) ) ) )
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
  (count preferenceA:red:blue_pyramid_block)
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
            (exists (?c - cube_block ?q - dodgeball)
              (exists (?f - dodgeball)
                (and
                  (and
                    (not
                      (and
                        (in_motion ?q)
                        (not
                          (in_motion ?f)
                        )
                        (agent_holds ?q ?f)
                        (above ?f rug)
                      )
                    )
                    (>= (distance room_center ?q) (distance ?q ?q))
                    (not
                      (in_motion ?q rug)
                    )
                  )
                  (in ?f)
                  (< 1 (distance room_center agent))
                  (agent_holds top_drawer ?f)
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
      (exists (?j - game_object)
        (then
          (once (and (not (in_motion ?j) ) (forall (?n - cube_block) (in_motion ?j ?j) ) ) )
          (once (same_color ?j ?j) )
          (once (not (in bed ?j) ) )
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
        (once (exists (?w - curved_wooden_ramp ?w - beachball) (not (and (on ?w) (rug_color_under desk) (in floor) ) ) ) )
        (hold (not (and (and (object_orientation ?xxx ?xxx) (or (agent_holds rug ?xxx) (on ?xxx) ) (and (on ?xxx) (not (in_motion ) ) ) ) (agent_holds west_wall) (in_motion ?xxx ?xxx) (>= 0 1) ) ) )
        (once (and (on ?xxx) (and (on ?xxx) (on ?xxx ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (>= (count preferenceB:blue_pyramid_block:blue_cube_block) (count-once-per-objects preferenceA:pink_dodgeball:book) )
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
              (forall (?h - hexagonal_bin ?r - (either cylindrical_block pencil))
                (agent_holds ?r ?r)
              )
            )
            (on ?xxx brown)
            (not
              (rug_color_under agent)
            )
            (in_motion ?xxx ?xxx)
            (object_orientation ?xxx)
            (faces ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?t ?d - game_object)
        (then
          (once (and (and (not (is_setup_object ?t ?t) ) (same_color ?t) (touch brown) ) (not (agent_holds ?t) ) ) )
          (hold (agent_holds agent) )
          (once (not (not (in_motion ?t ?d) ) ) )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (* 100 3 )
      (* (count preferenceA:blue_cube_block) (* (count-once-per-objects preferenceA:hexagonal_bin) 1 )
        (count preferenceA:pink)
      )
    )
    (and
      (>= (* (count preferenceA:golfball) (count preferenceA:red) )
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
        (hold (game_over ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
        (once (and (< (distance 5 room_center) 1) (not (touch ?xxx) ) (not (open upside_down ?xxx ?xxx) ) (and (in_motion ?xxx agent) (not (on ?xxx ?xxx) ) ) (in ?xxx bed) (exists (?r - teddy_bear ?y - building ?u - ball) (in_motion ?u ?u) ) (in_motion bed agent) ) )
      )
    )
    (preference preferenceB
      (then
        (hold (on ?xxx) )
        (hold (and (in_motion ?xxx) (agent_holds ?xxx) (and (and (not (agent_holds ?xxx) ) (not (in ?xxx ?xxx) ) ) (and (not (agent_holds ?xxx ?xxx) ) (exists (?k - ball ?s - red_pyramid_block ?w ?y - (either mug yellow_cube_block)) (same_color ?w) ) (on ?xxx agent) ) ) (and (touch floor) (agent_holds agent) ) (agent_holds ?xxx) (agent_holds agent) ) )
        (hold (not (agent_holds agent ?xxx) ) )
      )
    )
  )
)
(:terminal
  (> (count preferenceB:yellow:yellow) (- 10 )
  )
)
(:scoring
  (* 5 (or 10 (count-once-per-objects preferenceB:dodgeball) ) )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?c - drawer ?a - hexagonal_bin)
    (and
      (game-conserved
        (agent_holds ?a ?a)
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - pillow)
      (and
        (preference preferenceA
          (exists (?m - (either cube_block golfball))
            (exists (?g - (either golfball yellow_cube_block))
              (exists (?e - cube_block)
                (exists (?l - (either hexagonal_bin golfball key_chain))
                  (exists (?y - hexagonal_bin ?w - green_triangular_ramp ?v - chair ?t - blue_cube_block)
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
  (> (count preferenceA:beachball) (* 20 (count-once-per-objects preferenceA:pink) )
  )
)
(:scoring
  (count preferenceA:red)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?u - golfball)
    (forall (?j - game_object)
      (game-conserved
        (agent_holds ?u ?u)
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - dodgeball)
      (and
        (preference preferenceA
          (exists (?y - tall_cylindrical_block ?w ?n - dodgeball)
            (then
              (once (and (in_motion ?q ?q) (agent_holds front) ) )
              (hold-to-end (not (in_motion ?n ?w) ) )
              (once (not (and (exists (?o ?v - block) (agent_holds ?v) ) (and (not (in_motion rug) ) (and (in_motion agent ?n) (adjacent ?q floor) ) ) ) ) )
            )
          )
        )
        (preference preferenceB
          (exists (?b - hexagonal_bin)
            (exists (?m - dodgeball ?n ?k ?d ?s ?p ?f - hexagonal_bin)
              (at-end
                (and
                  (in_motion sideways ?b)
                  (in_motion ?k ?p)
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?d - ball ?r - cube_block)
        (exists (?o - (either triangular_ramp))
          (then
            (once (same_color agent) )
            (once (agent_holds ?o) )
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
      (exists (?e - chair ?t - (either pyramid_block book laptop))
        (forall (?x - cube_block ?y - teddy_bear)
          (and
            (game-conserved
              (and
                (agent_holds ?y)
                (agent_holds ?y ?t)
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
                    (exists (?o - doggie_bed ?q - block ?g - hexagonal_bin)
                      (game-conserved
                        (< 2 (distance 5 agent))
                      )
                    )
                    (exists (?m - shelf)
                      (exists (?q - hexagonal_bin)
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
                  (forall (?a - hexagonal_bin)
                    (exists (?t - cube_block ?r ?u ?z - cube_block)
                      (and
                        (exists (?j - game_object)
                          (forall (?m - game_object)
                            (and
                              (and
                                (and
                                  (game-optional
                                    (in_motion ?m)
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
      (exists (?q - game_object)
        (exists (?u - hexagonal_bin ?u - hexagonal_bin ?v - cube_block ?a - building)
          (then
            (any)
            (hold (agent_holds ?a) )
            (once (= (distance ?a ?a) (distance ?q ?q)) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-measure preferenceA:brown) (* 100 5 )
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
        (hold (and (in ?xxx ?xxx) (and (same_color back agent) (and (agent_holds ?xxx) (exists (?y - block) (agent_holds pink_dodgeball) ) (in_motion ?xxx ?xxx) ) ) ) )
        (hold (equal_x_position desk ?xxx) )
      )
    )
    (preference preferenceB
      (forall (?g - (either basketball key_chain) ?p - chair ?l - (either cylindrical_block golfball teddy_bear))
        (at-end
          (in_motion ?l)
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
        (count preferenceB:doggie_bed)
      )
    )
    (+ (+ (count preferenceA:dodgeball) (count-once preferenceB:dodgeball) (count-shortest preferenceB:dodgeball) )
      2
    )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?p - dodgeball ?s - cube_block)
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
        (exists (?r - cube_block)
          (exists (?c - pillow)
            (then
              (once (in ?r) )
              (hold (on ?b) )
              (once (not (in_motion ?r ?c) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preferenceA:blue_cube_block) (* (count-once preferenceA:top_drawer) (and (* (* (count-once preferenceA:green) (count preferenceA:alarm_clock) )
        )
        2
        (count preferenceA:side_table)
      )
      (count preferenceA:dodgeball:dodgeball)
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
    (forall (?p - (either lamp triangle_block tall_cylindrical_block) ?i - wall ?s - (either dodgeball laptop) ?d - block ?e - cube_block)
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
      (exists (?r - golfball)
        (exists (?j - ball)
          (at-end
            (on ?r)
          )
        )
      )
    )
    (preference preferenceB
      (exists (?w - golfball)
        (exists (?u ?n ?c - triangular_ramp)
          (exists (?i ?q - cube_block ?d - hexagonal_bin ?t - ball ?q - cube_block)
            (at-end
              (agent_holds ?n yellow)
            )
          )
        )
      )
    )
    (preference preferenceC
      (then
        (once (and (adjacent_side ?xxx ?xxx ?xxx) (agent_holds ?xxx) ) )
        (hold (= 0 (distance_side ?xxx ?xxx)) )
      )
    )
  )
)
(:terminal
  (>= 15 (+ (* (count preferenceC:dodgeball) (- 2 )
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
    (forall (?d - desk_shelf)
      (and
        (preference preferenceA
          (then
            (once (in bed) )
            (hold-to-end (and (adjacent rug) (not (not (and (exists (?s ?z - cube_block ?z - hexagonal_bin) (in ?d) ) (on ?d ?d) (and (agent_holds ?d rug) (agent_holds ?d) ) ) ) ) (< (distance 10 ?d) (distance agent)) ) )
            (once (in_motion ?d ?d) )
          )
        )
        (preference preferenceB
          (exists (?y - cube_block ?x - dodgeball)
            (exists (?y - hexagonal_bin ?j - building ?b - hexagonal_bin ?t - hexagonal_bin)
              (then
                (once (and (= 1 (distance ?d 10)) (in_motion ?x ?d) (and (and (not (in_motion ?d) ) (not (agent_holds ?x) ) ) (on ?x) ) ) )
                (once (or (not (adjacent ?d) ) (adjacent brown) ) )
                (once (and (adjacent_side ?d ?d) (<= 6 1) ) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (not (count-total preferenceA:beachball:dodgeball) ) (count preferenceA:yellow_cube_block) )
)
(:scoring
  (total-score)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (exists (?y ?q ?k - tan_cube_block ?n - curved_wooden_ramp)
      (exists (?g - dodgeball)
        (forall (?b - curved_wooden_ramp ?v - (either mug flat_block dodgeball))
          (and
            (and
              (game-optional
                (and
                  (and
                    (in_motion top_shelf ?v)
                    (in_motion ?n ?g)
                  )
                  (in_motion ?n ?v)
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
    (forall (?w - ball ?c ?n - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?k - building ?j - building ?d - dodgeball)
            (exists (?h - triangular_ramp ?x ?t ?r - chair ?m - (either main_light_switch alarm_clock credit_card blue_cube_block))
              (exists (?i - (either bridge_block book))
                (exists (?g - hexagonal_bin ?r - doggie_bed)
                  (exists (?w - game_object ?f - hexagonal_bin)
                    (exists (?s - hexagonal_bin)
                      (exists (?p ?h ?y - shelf)
                        (then
                          (hold-to-end (agent_holds ?p) )
                          (hold-while (between ?h) (rug_color_under agent) )
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
        (forall (?c - block)
          (and
            (forall (?u - game_object)
              (exists (?j - block ?y - building)
                (game-conserved
                  (not
                    (agent_holds ?c)
                  )
                )
              )
            )
            (forall (?s - block)
              (exists (?y - chair ?z - ball ?v - blinds)
                (game-conserved
                  (in ?c)
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
              (toggled_on ?xxx)
              (not
                (and
                  (exists (?j - cube_block)
                    (rug_color_under ?j)
                  )
                  (above ?xxx)
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
    (forall (?i - dodgeball)
      (and
        (preference preferenceA
          (exists (?y - block)
            (exists (?g - dodgeball)
              (exists (?u - hexagonal_bin ?p - dodgeball)
                (exists (?f - hexagonal_bin ?t - bridge_block)
                  (exists (?f - (either dodgeball golfball pyramid_block))
                    (exists (?u - (either pyramid_block basketball dodgeball))
                      (exists (?v - game_object)
                        (exists (?m - dodgeball)
                          (then
                            (hold (on ?u) )
                            (once (in_motion ?y) )
                            (once (in_motion bed ?t ?m) )
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
            (once (and (not (in_motion ?i ?i bed) ) (in_motion bed) (in_motion ?i) (exists (?c - (either triangle_block pyramid_block)) (not (in ?c ?i) ) ) (in rug desk) (and (on ?i ?i) (in_motion ?i) ) ) )
            (once (not (agent_holds ?i) ) )
            (once (adjacent_side ?i) )
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
        (once (exists (?l - hexagonal_bin) (in ?l) ) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count preferenceA:yellow:purple:side_table) 2 )
)
(:scoring
  (count preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (and
    (forall (?n - building ?l - block)
      (and
        (not
          (and
            (game-conserved
              (in_motion ?l ?l)
            )
          )
        )
        (exists (?e - game_object ?b ?y - wall)
          (game-conserved
            (not
              (adjacent ?l)
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
        (once (and (in_motion ?xxx) (in_motion ?xxx) (and (exists (?o - (either yellow_cube_block pencil) ?p - blinds) (and (agent_holds ?p) (not (and (and (and (< 1 1) ) (not (agent_holds ?p ?p) ) ) (adjacent ?p) ) ) ) ) (in_motion ?xxx) ) (adjacent_side ?xxx) (= (distance ) (distance ?xxx 8)) (< (distance door ?xxx ?xxx) (distance ?xxx ?xxx)) (same_color ?xxx) (and (= 1 7) (on bed ?xxx) ) ) )
        (once (game_over agent) )
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
    (exists (?s ?y - bridge_block)
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
    (forall (?g - hexagonal_bin ?p - chair)
      (and
        (preference preferenceA
          (then
            (hold-for 8 (and (< (distance ?p 2) 4) (agent_holds ?p) ) )
            (once (in_motion ?p ?p) )
            (once (agent_holds ?p) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (+ (+ (count-once-per-objects preferenceA:blue_pyramid_block) (count preferenceA:hexagonal_bin) )
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
            (count preferenceA)
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
      (exists (?m - cube_block)
        (game-optional
          (open ?m ?m)
        )
      )
      (and
        (game-conserved
          (and
            (in_motion ?xxx ?xxx)
            (forall (?f - red_dodgeball)
              (agent_holds east_sliding_door)
            )
          )
        )
      )
      (forall (?g - (either golfball cellphone))
        (game-conserved
          (agent_holds ?g ?g)
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
      (exists (?g - block ?j - cube_block)
        (then
          (hold (in_motion ?j) )
          (hold (not (agent_holds ?j) ) )
          (hold-to-end (and (agent_holds back green_golfball ?j) (on color rug) ) )
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
    (>= (count preferenceB:dodgeball) (- (count-once-per-objects preferenceA:golfball) )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?l - hexagonal_bin)
    (or
      (exists (?f - cube_block ?y - hexagonal_bin)
        (game-optional
          (agent_holds agent ?y)
        )
      )
      (game-conserved
        (touch ?l agent)
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
    (forall (?y - hexagonal_bin ?d ?e ?s ?y ?x ?t - hexagonal_bin)
      (and
        (preference preferenceB
          (then
            (hold (in_motion ?d) )
            (once (adjacent pink_dodgeball) )
            (once (touch ?e ?e) )
          )
        )
        (preference preferenceC
          (then
            (hold (on ?s ?s) )
            (once (agent_holds ?y) )
            (hold-while (not (in_motion ?x) ) (on ?t) )
          )
        )
        (preference preferenceD
          (exists (?g - ball)
            (then
              (once (< 0.5 0) )
              (once (and (touch ?y ?t) (in_motion ?x) ) )
              (hold-while (and (not (and (not (touch ?s) ) (touch ?g) ) ) (not (touch ?x) ) ) (and (< (distance ?s 9) 1) (agent_holds bed) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- (* (count preferenceB:dodgeball) (+ (count-measure preferenceA:dodgeball:green) 3 )
      )
    )
    (count preferenceB:basketball:hexagonal_bin:red)
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
  (forall (?g - hexagonal_bin)
    (game-conserved
      (and
        (and
          (and
            (< 2 (distance desk ?g))
          )
          (and
            (agent_holds ?g bed)
          )
        )
        (< (distance ?g 0) (distance agent ?g ?g))
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?b - doggie_bed)
        (exists (?m - curved_wooden_ramp)
          (exists (?y - hexagonal_bin ?i - hexagonal_bin ?y - game_object)
            (then
              (hold (and (touch ?m) (in_motion ?y) ) )
              (hold (not (agent_holds desk ?b) ) )
              (hold-while (on upright ?m) (not (not (and (adjacent ?m bridge_block) (not (not (or (and (exists (?h - (either cylindrical_block teddy_bear)) (and (not (agent_holds ?m) ) (touch ?y) (in ?b agent) ) ) (and (or (in ?m ?m) (in_motion ?m) ) (in_motion pink) ) (on desk) (on ?m) ) (in ?y) (in_motion ?b desk) (not (and (in_motion rug floor) (in_motion ?m) (not (agent_holds ?y ?b ?b ?b) ) ) ) ) ) ) (and (in_motion ?b brown) (agent_holds ?y) ) ) ) ) )
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
    (>= (* (count preferenceB:hexagonal_bin) 3 )
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
        (forall-sequence (?i - red_pyramid_block)
          (then
            (once (in ?i agent) )
            (once (not (and (in ) (agent_holds ?i) ) ) )
            (hold-while (agent_holds ?i) (not (in_motion ?i) ) )
          )
        )
        (once (object_orientation ?xxx) )
        (once (in_motion ?xxx) )
      )
    )
    (preference preferenceB
      (exists (?l - (either dodgeball floor))
        (exists (?s - (either dodgeball tall_cylindrical_block) ?b - (either cellphone top_drawer) ?g - (either book floor) ?y ?c - doggie_bed ?i - teddy_bear)
          (exists (?e - (either yellow_cube_block cellphone))
            (exists (?o - hexagonal_bin)
              (then
                (hold-while (not (< 1 1) ) (not (not (touch ?i ?e) ) ) )
                (hold (agent_holds ?o) )
                (hold (not (above ?i rug) ) )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (then
        (hold (same_color ?xxx ?xxx) )
        (once (and (and (in floor) (on ?xxx ?xxx) ) (game_start ?xxx ?xxx bed) (in agent ?xxx) ) )
        (once (and (and (and (in_motion ?xxx) (and (adjacent floor ?xxx) (not (and (not (and (not (< (distance side_table ?xxx) 6) ) (not (not (and (not (on ?xxx ?xxx) ) (not (and (on floor ?xxx) (in_motion ?xxx) (and (and (forall (?s - doggie_bed) (in_motion ?s pink_dodgeball) ) (on ?xxx ?xxx) (agent_holds ?xxx) ) (in_motion ?xxx) ) (not (agent_holds ?xxx ?xxx ?xxx) ) ) ) (in_motion ?xxx ?xxx) ) ) ) ) ) (agent_holds ?xxx) ) ) (in_motion ?xxx agent) (not (exists (?p - hexagonal_bin ?s - (either blue_cube_block ball)) (in_motion ?s ?s) ) ) (adjacent_side ?xxx ?xxx) (and (agent_holds ?xxx) (exists (?l - bridge_block ?y - watch) (in_motion agent ?y) ) (on ?xxx) ) ) (and (adjacent ?xxx) (< 2 (distance ?xxx room_center)) ) (on ?xxx bridge_block) ) (and (on ?xxx) (touch ?xxx door) ) ) (agent_holds ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (> (* (count-once preferenceC:blue_dodgeball) (count preferenceB:triangle_block) )
      (count preferenceB:pink)
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
  (forall (?d - cube_block)
    (game-conserved
      (in_motion ?d west_wall)
    )
  )
)
(:constraints
  (and
    (forall (?s - hexagonal_bin ?b - curved_wooden_ramp)
      (and
        (preference preferenceA
          (then
            (once (forall (?h - doggie_bed) (in_motion ?h ?h) ) )
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
    (exists (?a - hexagonal_bin)
      (game-conserved
        (on ?a ?a)
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
    (forall (?z ?s - color)
      (and
        (preference preferenceA
          (exists (?p ?o - triangular_ramp ?u - curved_wooden_ramp)
            (then
              (once (and (exists (?f - (either cube_block)) (and (not (agent_holds ?u) ) (and (exists (?n - block) (same_object ?s) ) (not (rug_color_under desk) ) (in_motion ?s ?z) (in_motion ?f ?z) (same_type ?u ?f) (on ?u) (and (not (and (not (equal_z_position ?f) ) (and (and (not (>= 6 (distance ?u ?u)) ) (is_setup_object rug) (or (on ?z) (in ?f) ) (and (and (in_motion ?s pink) (adjacent_side ?f) ) (not (not (on ?u) ) ) ) (in_motion ?f) (object_orientation ?s) ) (and (in_motion ?z) (in ?u) ) ) ) ) (not (same_object ?f ?z) ) ) (not (and (not (agent_holds ?z) ) (agent_holds ?s) ) ) ) ) ) (< 1 7) ) )
              (hold (agent_holds ?u) )
              (once (agent_holds desk ?s) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (in_motion ?xxx ?xxx) )
        (hold (or (not (game_start ?xxx ?xxx) ) (not (in_motion ?xxx ?xxx) ) (< (distance room_center ?xxx) (distance desk ?xxx)) ) )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (<= 4 (count preferenceB:dodgeball:tan) )
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
    (< (count-once-per-objects preferenceA:rug) (* (count preferenceA:dodgeball) (count preferenceA:green:blue_pyramid_block) )
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
    (forall (?c - ball)
      (and
        (preference preferenceA
          (at-end
            (agent_holds )
          )
        )
        (preference preferenceB
          (exists (?r - pillow ?k ?b - dodgeball)
            (then
              (once (touch ?c) )
              (once (and (not (> (distance ?b ?b) 1) ) (not (not (agent_holds ?b ?b) ) ) ) )
              (once (agent_holds ?b ?k) )
            )
          )
        )
        (preference preferenceC
          (then
            (hold (in ?c ?c) )
            (once (not (faces ?c ?c) ) )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?x - chair ?g - building ?g - cube_block)
        (at-end
          (not
            (agent_holds ?g)
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
            (* (and (count preferenceB) ) (count preferenceA:pink) )
          )
          (count-once preferenceA:dodgeball)
          5
        )
      )
      (* (and (count preferenceD:purple) (* (count-longest preferenceB:pink_dodgeball) (count preferenceA:pink_dodgeball) )
          (+ (count preferenceC:blue_dodgeball:dodgeball) (- (total-time) )
            (= (- (count preferenceB:yellow) )
            )
          )
        )
        30
        4
        8
        (count preferenceB:yellow)
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
          (>= (count-once-per-objects preferenceC:dodgeball:yellow_pyramid_block) (count preferenceB:book) )
          (> (+ (count-once-per-objects preferenceB:dodgeball) (= (+ (count-overlapping preferenceB) (count preferenceD:beachball) )
                30
                (count preferenceA:golfball)
              )
            )
            (<= (<= 1 (count preferenceD:red) )
              (external-forall-maximize
                (count preferenceB:green:green)
              )
            )
          )
        )
        (< (count preferenceC:tan) 10 )
        (and
          (>= 5 (count preferenceC:alarm_clock) )
        )
      )
      (>= (count-once-per-objects preferenceB:yellow_cube_block) (* (count preferenceD:blue_pyramid_block) (count preferenceA:beachball) )
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
          (hold (forall (?h - cube_block ?m - hexagonal_bin ?i - (either golfball dodgeball)) (and (in_motion ?e) (agent_holds ?e ?e) ) ) )
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
  (exists (?x ?z - dodgeball)
    (game-conserved
      (on ?x)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?g - cube_block)
        (exists (?s - hexagonal_bin ?z - block)
          (then
            (once (and (agent_holds ?g) (not (and (in ?g ?g) (on desk ?g) (same_color ?z) (agent_holds ?z ?z) ) ) ) )
            (once (not (not (and (on ?z) (and (and (not (< 1 (distance ?g desk)) ) (on ?g ?g) ) (not (above desk) ) ) ) ) ) )
            (once (not (in_motion ?g ?z) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (> (* 100 (+ (* (count preferenceA:block) (count-once-per-objects preferenceA:doggie_bed) )
        (* 50 (count preferenceA:pink) )
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
    (count preferenceA:golfball)
    (* (= (count preferenceA:pink_dodgeball:dodgeball) (count preferenceA:pink) )
      (count preferenceA:pink)
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
    (forall (?y ?r - hexagonal_bin)
      (and
        (preference preferenceA
          (exists (?d - ball)
            (then
              (once (and (on pillow ?d) (touch ?y rug) ) )
              (once (agent_holds ?d) )
            )
          )
        )
      )
    )
    (forall (?q - tall_cylindrical_block)
      (and
        (preference preferenceB
          (exists (?j - (either dodgeball golfball) ?r - cube_block)
            (then
              (hold (agent_holds ?q) )
              (once (agent_holds ?q) )
              (hold (on ?q) )
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
      (exists (?m - wall ?b - ball)
        (exists (?u ?k - hexagonal_bin ?e - (either tall_cylindrical_block book golfball blue_cube_block dodgeball floor key_chain))
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
      (exists (?m - game_object ?e - shelf)
        (exists (?l - cube_block)
          (exists (?x - ball)
            (at-end
              (in_motion ?l ?x)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= 18 (+ (- (total-time) )
      (+ (* (count preferenceA:beachball:golfball) (* 0 )
        )
        (count-once preferenceA:blue_dodgeball)
      )
    )
  )
)
(:scoring
  (+ (- (external-forall-maximize 4 ) )
    (count preferenceA)
    15
    (* (count preferenceA:beachball) (count-once-per-objects preferenceA:green) (* 6 100 )
    )
    (total-time)
    (count-once-per-objects preferenceA:red:green)
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?u - hexagonal_bin ?j - triangular_ramp)
    (forall (?y - triangular_ramp ?o - (either rug rug))
      (exists (?a - (either yellow yellow))
        (game-conserved
          (agent_holds ?j ?o)
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
    (forall (?q - (either cube_block golfball cylindrical_block game_object blue_cube_block dodgeball blue_cube_block))
      (and
        (preference preferenceB
          (then
            (once (agent_holds ?q ?q) )
            (hold (agent_holds ?q) )
            (hold (and (in_motion floor) (faces bed ?q) ) )
          )
        )
        (preference preferenceC
          (exists (?k - game_object)
            (exists (?d - (either pillow pyramid_block))
              (exists (?x - beachball)
                (exists (?h - building)
                  (then
                    (once (not (not (and (on ?q ?d) ) ) ) )
                    (once (and (agent_holds ?h) (in_motion ?x) ) )
                    (once (not (not (not (> (distance desk 0 ?q) (distance ?x green_golfball)) ) ) ) )
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
        (hold (adjacent_side ?xxx) )
        (once (touch ?xxx) )
      )
    )
    (preference preferenceE
      (exists (?v - sliding_door)
        (then
          (hold (in_motion ?v) )
          (once (adjacent ?v ?v) )
          (once (and (and (same_color ?v south_west_corner) (is_setup_object ?v ?v) (= (distance room_center ?v) (distance ?v) 1 2) ) (agent_holds ?v) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (* (count-once-per-objects preferenceA:pink) (count preferenceE:purple:yellow) )
        (count-once-per-objects preferenceB:beachball)
      )
      (count preferenceC:yellow)
    )
    (<= (count-once-per-objects preferenceE:triangle_block) (* (* (- (+ 9 (- 0 )
            )
          )
          (< (count preferenceC:dodgeball) 3 )
        )
        (- (count preferenceB:brown:beachball) )
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
    (forall (?c - dodgeball)
      (and
        (preference preferenceA
          (exists (?p - game_object)
            (then
              (hold-while (is_setup_object ?c) (not (not (< (distance room_center ?p) (distance front_left_corner agent)) ) ) )
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
    (< (count preferenceA:yellow) (* (* 15 (count preferenceA:doggie_bed) (external-forall-maximize (count-once-per-external-objects preferenceA:pink) ) )
        (count-once-per-objects preferenceA:dodgeball)
        (count preferenceA:alarm_clock)
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
        (exists (?y - hexagonal_bin)
          (then
            (once (and (touch ?y) (and (not (in_motion ?y floor) ) (in ?b agent) (and (and (agent_holds ?b ?y) (in_motion ?b) ) (in_motion ?b) (and (in_motion ?b) (not (agent_holds ?y) ) (on ?y ?y) (and (adjacent_side ?b ?y) (and (not (not (on ?b) ) ) (not (on ?y ?b) ) ) ) (in_motion ?b) ) (exists (?s - color) (in ?y) ) ) ) (and (agent_holds ?b ?b) (in ?b ?b) ) (on ?y) ) )
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
      (exists (?j - teddy_bear)
        (exists (?a - cube_block)
          (then
            (hold (not (not (in ?a) ) ) )
            (hold-while (forall (?r - hexagonal_bin) (in_motion ?j ?j) ) (not (in_motion ?a) ) (and (on ?j) (and (or (and (not (in_motion ?a agent) ) (and (and (not (agent_holds ?j) ) (in_motion front) ) (not (agent_holds ?j) ) ) ) (on ?a ?j ?a) ) (in_motion ?a) ) ) )
            (hold (not (and (agent_holds ?a) (and (and (agent_holds ?a ?j) (not (in_motion ?a) ) ) (and (and (not (in_motion ?j floor) ) (not (in_motion ?j ?j) ) ) (on ?a) ) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 6 (* (+ 50 (count preferenceB:dodgeball) )
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
      (exists (?h - teddy_bear ?s - color)
        (exists (?n - (either pyramid_block pink pyramid_block wall wall flat_block ball) ?d - ball)
          (exists (?z - hexagonal_bin ?t - hexagonal_bin)
            (game-conserved
              (in_motion ?d ?d)
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
      (exists (?u ?a - ball ?m - (either mug cellphone golfball) ?j - hexagonal_bin)
        (then
          (hold (agent_holds ?j) )
          (once (and (and (forall (?z ?v - rug) (not (in ?z desk) ) ) (on bed) ) (in_motion desk ?j) ) )
          (hold (touch ?j) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 3 (count-once-per-objects preferenceA:pink) )
    (>= (count preferenceA:golfball) (count preferenceA:pink) )
    (or
      (not
        (>= 1 (count-once preferenceA:top_drawer:hexagonal_bin) )
      )
      (>= 4 (* (count preferenceA:triangle_block) (count preferenceA:dodgeball:basketball) )
      )
    )
  )
)
(:scoring
  (count preferenceA:golfball)
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
    (forall (?w - block)
      (and
        (preference preferenceB
          (exists (?y - color)
            (exists (?b - ball)
              (exists (?x - shelf)
                (then
                  (once (on ?x) )
                  (once (and (and (not (not (and (and (adjacent ?b) (in_motion ) ) (exists (?l - ball) (in_motion ?y agent) ) ) ) ) (not (exists (?d - dodgeball) (and (and (adjacent_side bed) (agent_holds ?x bed) ) (agent_holds ?d door) (agent_holds ?y) ) ) ) ) (on ?x) ) )
                  (hold (in_motion desk) )
                )
              )
            )
          )
        )
      )
    )
    (preference preferenceC
      (exists (?t - hexagonal_bin ?l - doggie_bed)
        (exists (?x - dodgeball ?z - chair)
          (then
            (once (not (and (broken ?z upside_down) (not (in_motion upright) ) ) ) )
            (once (same_color ?l ?l) )
            (hold (in_motion ?l) )
          )
        )
      )
    )
    (preference preferenceD
      (exists (?d - hexagonal_bin)
        (then
          (once (agent_holds ?d ?d) )
          (once (in_motion ?d) )
          (once (in_motion ?d) )
        )
      )
    )
    (preference preferenceE
      (exists (?x - cylindrical_block ?y - dodgeball)
        (at-end
          (not
            (on ?y)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preferenceA:golfball:dodgeball) (not (+ (count-same-positions preferenceB) (count-unique-positions preferenceC:block:doggie_bed) )
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
  (>= (count-overlapping preferenceA:basketball) (count preferenceA:basketball) )
)
(:scoring
  (count-once preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?u - desk_shelf)
    (game-optional
      (on bed agent)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?y - hexagonal_bin)
        (exists (?g - game_object ?v - (either cube_block cube_block) ?v - hexagonal_bin ?l - triangular_ramp)
          (exists (?j - color ?t - hexagonal_bin)
            (then
              (once (not (object_orientation ?l ?t) ) )
              (once (in ?l) )
              (once (not (same_object ?y) ) )
            )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?s - wall)
        (exists (?r - dodgeball)
          (then
            (hold (on ?r ?s) )
            (once (and (and (in ?s) (not (in_motion ?s) ) ) (in_motion ?s ?r) ) )
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
                  (* (* (count preferenceA:pink) (* (count-shortest preferenceB) (* (count-once-per-objects preferenceA:alarm_clock) (external-forall-maximize (count-once-per-objects preferenceA:basketball) ) )
                      )
                    )
                    (count preferenceA:hexagonal_bin)
                  )
                )
                (* 10 10 )
              )
            )
            (* (* (count-once-per-objects preferenceB:red:doggie_bed) 2 )
              6
              (= (+ (count-once-per-objects preferenceB:dodgeball) (count preferenceB:golfball) )
                (+ 8 7 )
                (count-once-per-objects preferenceB:red)
              )
            )
            (count preferenceA:golfball)
          )
          (<= (count-once-per-objects preferenceB:hexagonal_bin) (count preferenceA:tan:beachball) )
        )
        (- 7 )
      )
    )
    (external-forall-minimize
      (count preferenceA:beachball)
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
      (exists (?u - block ?u - pyramid_block)
        (then
          (hold-while (on ?u) (in_motion bed) )
          (hold (not (not (in_motion desk) ) ) )
          (once (or (not (touch ?u) ) (on ?u) (and (adjacent ?u) (and (toggled_on desk) (not (or (not (and (agent_holds ?u) (and (agent_holds ?u ?u) (and (or (not (and (on ?u ?u) (and (agent_holds ?u ?u) (and (and (or (on ?u ?u) (touch ?u) ) (> 2 1) ) (not (in_motion ?u ?u) ) ) (agent_holds ?u) ) ) ) (in_motion ?u) ) (not (agent_holds ?u ?u) ) (not (adjacent_side ?u) ) ) ) ) ) (not (agent_holds ?u ?u) ) ) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (count preferenceA:beachball) 0 )
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
    (= (count preferenceA:golfball) 3 )
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
    (forall (?v - beachball ?i - teddy_bear)
      (and
        (preference preferenceA
          (exists (?r - dodgeball)
            (exists (?n - teddy_bear)
              (forall (?k - (either floor pencil dodgeball))
                (then
                  (once-measure (adjacent ?k ?n) (distance ?i 7) )
                  (once (and (agent_holds ?i) (and (on ?n) (not (and (agent_holds ?k) (agent_holds ?i) ) ) ) ) )
                  (once (not (not (on top_drawer ?i) ) ) )
                )
              )
            )
          )
        )
        (preference preferenceB
          (then
            (hold-while (in_motion ?i ?i) (on ?i ?i) )
            (once-measure (agent_holds ?i desk) (distance ?i desk) )
            (once (agent_holds ?i ?i) )
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
        (count preferenceB:brown)
      )
    )
    7
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?h - golfball)
    (game-optional
      (in_motion ?h ?h)
    )
  )
)
(:constraints
  (and
    (forall (?j - color)
      (and
        (preference preferenceA
          (exists (?u - (either tall_cylindrical_block alarm_clock))
            (then
              (hold (not (not (not (touch ?j agent) ) ) ) )
              (once (agent_holds ?j ?u) )
              (once (agent_holds ?u ?u) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count preferenceA:basketball:orange) 20 )
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
  (* (count preferenceA:beachball:red:green) (* (count-overlapping preferenceB:cube_block) (count-once preferenceA:basketball) )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?o - ball)
    (and
      (exists (?e - hexagonal_bin)
        (and
          (game-conserved
            (on ?e ?o)
          )
          (game-conserved
            (adjacent_side ?o)
          )
          (and
            (and
              (or
                (game-conserved
                  (adjacent ?o)
                )
                (game-conserved
                  (in_motion agent)
                )
                (game-conserved
                  (on ?e ?e)
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
  (= (- (* (count preferenceA:basketball) (count-longest preferenceA:doggie_bed) (count-once-per-objects preferenceA:dodgeball) (- (* 2 15 )
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
              (count preferenceA:alarm_clock)
              (* 6 1 )
              (* (* (+ (count preferenceA:top_drawer) )
                  (* 2 )
                )
                (total-score)
              )
              (count preferenceA:basketball)
            )
            (count-once-per-objects preferenceA:beachball:dodgeball)
          )
          (* (* (= 1 6 )
              (+ 5 (* (+ (count-once-per-objects preferenceA:dodgeball) (count-once-per-objects preferenceA:beachball:golfball) (count preferenceA:hexagonal_bin:green) (count preferenceA:basketball:alarm_clock) (count-unique-positions preferenceA) (count preferenceA:dodgeball) )
                  5
                )
              )
            )
            (count-once-per-objects preferenceA:beachball)
            (count preferenceA:dodgeball)
          )
        )
        3
        (count preferenceA:hexagonal_bin)
        (count-increasing-measure preferenceA:yellow_pyramid_block)
        (+ (* (count preferenceA:beachball) (total-score) )
          (count-once-per-objects preferenceA:book)
          3
          (count preferenceA:basketball)
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
      (count preferenceA:red:dodgeball)
    )
    (count-once-per-objects preferenceA:beachball)
    (+ (external-forall-maximize (* (count preferenceA:pink_dodgeball:basketball) 1 )
      )
      (count preferenceA:beachball)
      (* 8 10 (or (* (total-time) (* (* 100 (- (= (* (count preferenceA:dodgeball) (count preferenceA:golfball:book) )
                    (external-forall-minimize
                      (+ (count-shortest preferenceA:dodgeball) (* (count preferenceA:yellow) 2 (+ (* (count-shortest preferenceA:basketball) (external-forall-maximize (count preferenceA:pink) ) )
                            (count preferenceA:basketball:red_pyramid_block:hexagonal_bin)
                          )
                        )
                        5
                        (count-measure preferenceA:basketball:doggie_bed)
                        (* (count preferenceA:red_pyramid_block:basketball) (count preferenceA:purple:bed:dodgeball) )
                        (+ 4 (* (- (count-longest preferenceA:dodgeball) (count preferenceA:red) ) (* (count-increasing-measure preferenceA:dodgeball) 1 )
                          )
                        )
                      )
                    )
                    (count-once preferenceA:pyramid_block:triangle_block)
                  )
                )
              )
              (* (* (* (count-overlapping preferenceA:beachball) (* (count preferenceA:brown) (count-once-per-objects preferenceA:dodgeball) )
                    (/
                      3
                      (+ 2 (count preferenceA:purple) )
                    )
                  )
                  (count-measure preferenceA:dodgeball:blue_dodgeball)
                  (+ (* 1 (count-once-per-objects preferenceA:basketball:white) (count-longest preferenceA:basketball) )
                    3
                  )
                )
                3
              )
            )
          )
          (count-once preferenceA:pink:hexagonal_bin:beachball)
          (* (* (+ 6 (- (* 4 (count preferenceA:beachball) 4 (+ (total-time) (+ (count-once preferenceA:dodgeball) 5 )
                    )
                    (count preferenceA:red)
                    2
                  )
                )
              )
              (* (count preferenceA:dodgeball:beachball) (+ (count-increasing-measure preferenceA:dodgeball) (count preferenceA:beachball) (count-once-per-objects preferenceA:yellow) (count preferenceA:orange) (count preferenceA:dodgeball:beachball) )
              )
            )
            (count-same-positions preferenceA:side_table)
          )
        )
        (count-overlapping preferenceA:triangle_block:doggie_bed)
        (+ 7 (* 30 (* (total-score) (+ (count-once-per-objects preferenceA:basketball) 0 )
              (* (count preferenceA:dodgeball) 5 )
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
  (exists (?u - (either main_light_switch cube_block))
    (forall (?f - hexagonal_bin)
      (game-optional
        (not
          (not
            (in ?f)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?k - block)
        (at-end
          (not
            (not
              (and
                (agent_holds ?k ?k)
                (>= (distance front) 2)
                (in ?k)
                (not
                  (not
                    (in_motion ?k ?k)
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
  (> 40 (* (count preferenceA:basketball:golfball) )
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
    (forall (?w - (either basketball dodgeball))
      (and
        (preference preferenceA
          (then
            (once (not (not (and (in ?w ?w) (and (and (agent_holds ?w ?w) (and (and (in_motion ?w) (in_motion ?w brown) ) (in_motion ?w) ) ) (agent_holds ?w) ) ) ) ) )
            (once (not (in_motion ?w ?w) ) )
            (once (agent_holds ?w) )
          )
        )
      )
    )
    (forall (?f - shelf ?h ?z - cylindrical_block)
      (and
        (preference preferenceB
          (then
            (forall-sequence (?l - game_object)
              (then
                (once (in ?l) )
                (once (and (and (not (in_motion ?z) ) (touch ?l) ) (in ?z) ) )
                (hold (not (in south_west_corner) ) )
              )
            )
            (once (or (agent_holds desk ?h) (not (exists (?v - teddy_bear ?y - building) (agent_holds ?z desktop) ) ) (adjacent ?h) (not (agent_holds ?z) ) ) )
            (hold (agent_holds ?h) )
          )
        )
        (preference preferenceC
          (then
            (once (exists (?w - game_object) (agent_holds ?h ?w) ) )
            (hold (agent_holds ?h ?z) )
            (once (agent_holds ?z) )
          )
        )
        (preference preferenceD
          (then
            (once (same_color ?z floor) )
            (once (object_orientation ?h) )
            (once (agent_holds ?z) )
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
      (exists (?m - game_object)
        (then
          (hold (not (and (agent_holds ?m ?m) (touch ?m ?m) ) ) )
          (once (agent_holds ?m ?m) )
          (once (and (in ?m ?m) (in_motion ?m) (on ?m ?m) ) )
          (hold (and (same_color ?m ?m) (in ?m) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* (* (* 2 (count-once-per-objects preferenceA:pink) )
          (count preferenceA:yellow)
        )
        (* 0 (count preferenceA:basketball) (total-score) )
      )
      3
    )
    (count-once-per-external-objects preferenceA:beachball)
  )
)
(:scoring
  (count preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (game-conserved
    (forall (?q - pyramid_block)
      (in_motion ?q agent)
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
                (forall (?k - drawer)
                  (agent_holds ?k)
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
          (exists (?u - dodgeball)
            (in_motion ?u ?u)
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preferenceA:book) (> (+ (count preferenceA:pink_dodgeball) 30 )
        (* (count-once-per-objects preferenceA:hexagonal_bin) (* 300 (total-score) )
        )
      )
    )
    (< (= (count preferenceA:dodgeball) 3 (= (* (count preferenceA:dodgeball:basketball) 5 )
          10
        )
      )
      (count-once preferenceA:top_drawer)
    )
  )
)
(:scoring
  (count preferenceA:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?n - hexagonal_bin ?e - hexagonal_bin)
    (and
      (forall (?w - hexagonal_bin)
        (exists (?m - (either game_object dodgeball))
          (game-conserved
            (not
              (adjacent_side ?e)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?n - hexagonal_bin ?z - game_object)
      (and
        (preference preferenceA
          (then
            (hold (not (not (agent_holds ?z ?z) ) ) )
            (once (not (and (and (in_motion ?z ?z) (not (not (same_type ?z ?z) ) ) ) (in_motion front ?z) ) ) )
            (hold-while (agent_holds ?z agent) (in_motion ?z) )
          )
        )
        (preference preferenceB
          (exists (?e - game_object)
            (then
              (hold-while (agent_holds main_light_switch) (not (agent_holds ?e ?e) ) )
              (once (in_motion ?z ?z) )
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
    (>= (count preferenceA:pink) 180 )
    (>= (count-unique-positions preferenceA:blue_dodgeball) (* (count preferenceB:basketball) (count preferenceB:dodgeball) (* 15 (+ 0 (count-increasing-measure preferenceB:dodgeball) 3 (- 20 )
            (* (external-forall-maximize (+ (* (* (* (* (count-once-per-objects preferenceA:pink_dodgeball) (count preferenceB:blue_dodgeball) )
                        30
                      )
                      (* (* (count preferenceB:red) (* (count preferenceB:basketball) (* 100 (count preferenceA:yellow_pyramid_block) )
                            3
                            2
                          )
                        )
                        (+ (count preferenceA:beachball) (count preferenceB:pink_dodgeball:red_pyramid_block) )
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
      (exists (?q ?u ?o - cube_block ?b - hexagonal_bin)
        (forall (?r ?v - ball)
          (game-conserved
            (and
              (open ?b ?v)
              (equal_z_position floor)
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
        (hold-to-end (not (adjacent_side ?xxx) ) )
        (any)
        (hold (and (not (not (and (and (on ?xxx) (in ?xxx ?xxx) (agent_holds door ?xxx) ) (in ?xxx ?xxx) ) ) ) (rug_color_under ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preferenceA:dodgeball) 8 )
    (> (count-once-per-objects preferenceA:pink:blue_pyramid_block) (* (count preferenceA:dodgeball) (count preferenceA:white) )
    )
    (>= (- 5 )
      100
    )
  )
)
(:scoring
  (* (* (count preferenceA:orange) (count preferenceA:pink) )
    (total-score)
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?p - dodgeball)
    (game-optional
      (> (distance ?p) (distance ?p desk))
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
    (<= (* (count-shortest preferenceA:beachball) (count preferenceA:dodgeball) 0 (count-once-per-external-objects preferenceA:yellow) (count preferenceA:dodgeball:dodgeball) 2 )
      (count-once-per-objects preferenceA:pink)
    )
    (= 10 (count-unique-positions preferenceA:dodgeball:basketball) )
  )
)
(:scoring
  (* (count preferenceA:dodgeball:hexagonal_bin) (count preferenceA:book) (count-increasing-measure preferenceA:pyramid_block) )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?n - (either basketball ball book cylindrical_block key_chain cellphone pen) ?q - (either laptop laptop) ?z - dodgeball)
    (game-optional
      (agent_holds ?z)
    )
  )
)
(:constraints
  (and
    (preference preferenceA
      (exists (?t - ball)
        (exists (?g - hexagonal_bin)
          (then
            (hold (and (and (equal_z_position ?t ?t) (on ?g) ) ) )
            (once (touch ?g) )
            (once (and (not (not (on ?t ?t) ) ) (and (not (in_motion ?t ?t) ) (and (and (and (exists (?p - game_object ?z - (either basketball basketball)) (and (not (and (on ?g ?z) (agent_holds ?t ?g) (in_motion ?g ?t) ) ) (in_motion ?z top_shelf) ) ) (not (game_over ?t ?t) ) (agent_holds pillow) ) (on ?g ?t) ) (in ?g upright) ) (adjacent ?g) ) (not (and (not (and (not (in ?t ?t) ) (and (in_motion ?g) (< 4 1) ) ) ) (and (adjacent ?t ?g) (in_motion ?g) ) ) ) ) )
          )
        )
      )
    )
    (preference preferenceB
      (exists (?x ?w - hexagonal_bin)
        (then
          (once (agent_holds rug ?x) )
          (hold (= 1 (distance ?x ?x) (distance_side ?x)) )
          (hold-while (and (< 2 (distance room_center agent)) (not (not (in_motion ?x ?x) ) ) ) (in_motion ?x ?w ?x) (touch ?x) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preferenceB:bed:dodgeball) 3 )
    (>= 8 (= 3 (* (+ 6 10 (count-once-per-external-objects preferenceB:pink) )
          (* 10 (= (total-time) )
          )
        )
      )
    )
    (>= (+ (count preferenceB:pink:beachball) 5 )
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
  (exists (?h - chair)
    (game-conserved
      (not
        (not
          (in_motion ?h)
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
          (once (in_motion ?i) )
          (hold (in_motion agent) )
          (once (same_type ?i) )
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
  (+ (count preferenceB:beachball) (* (count preferenceB:doggie_bed) )
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
      (exists (?t - hexagonal_bin)
        (then
          (once (and (on ?t door) (and (same_color ?t) (in_motion desk) (and (in_motion agent) (not (same_type ?t ?t) ) ) ) ) )
          (once (not (in ?t) ) )
          (hold-while (exists (?r - curved_wooden_ramp ?j ?i - (either chair cube_block book red)) (and (touch ?t) (and (and (in_motion ?i) (and (agent_holds ?j ?j) (in ?t) ) ) (agent_holds ?t ?i) ) (not (agent_holds ?i) ) ) ) (in ?t ?t) )
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
        (exists (?c - curved_wooden_ramp)
          (then
            (once (in_motion tan) )
            (once (not (agent_holds ?c ?b) ) )
            (once (or (agent_holds ?b ?c) (not (and (and (and (not (agent_holds ?b ?c) ) (not (and (agent_holds ?c ?b) (on ?b ?b) ) ) ) (not (adjacent ?c) ) ) (in_motion ?b ?b) ) ) (not (and (> 5 1) (and (agent_holds ?c ?c) (and (and (agent_holds bed ?b) (not (in_motion ?b ?c) ) (in_motion ) ) (exists (?j - curved_wooden_ramp ?a - (either dodgeball cylindrical_block)) (not (on ?a) ) ) ) ) (agent_holds ?c bed) ) ) (or (and (not (in agent ?b) ) (on ?c ?c) ) (agent_holds ?b ?c) ) ) )
          )
        )
      )
    )
    (preference preferenceB
      (then
        (once (not (agent_holds ?xxx) ) )
        (once (in_motion ?xxx) )
        (once (and (and (agent_holds ?xxx ?xxx) (not (not (not (not (not (forall (?j - (either teddy_bear golfball)) (adjacent ?j) ) ) ) ) ) ) ) (on ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (or
    (>= 18 (count preferenceB:dodgeball) )
    (> (count-overlapping preferenceA:dodgeball) (count-once preferenceA:purple:green) )
  )
)
(:scoring
  (count preferenceA:red:dodgeball)
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (forall (?n - hexagonal_bin)
    (forall (?i - (either bridge_block mug basketball) ?q - dodgeball)
      (and
        (exists (?t - pyramid_block ?m - doggie_bed)
          (and
            (and
              (exists (?y - hexagonal_bin)
                (forall (?u - hexagonal_bin ?b - beachball)
                  (game-conserved
                    (not
                      (agent_holds brown)
                    )
                  )
                )
              )
              (game-conserved
                (or
                  (in_motion ?q ?q)
                )
              )
            )
            (and
              (or
                (and
                  (exists (?d - hexagonal_bin)
                    (forall (?i - book ?f - teddy_bear)
                      (and
                        (exists (?r - building)
                          (and
                            (game-optional
                              (exists (?w - cube_block)
                                (in_motion ?d ?f)
                              )
                            )
                            (forall (?i - golfball)
                              (exists (?v ?o - dodgeball ?p - (either desktop))
                                (game-conserved
                                  (exists (?o - hexagonal_bin ?z - hexagonal_bin ?z - (either dodgeball doggie_bed) ?a - cube_block)
                                    (in_motion ?p ?i)
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
                            (in_motion ?n ?d)
                          )
                        )
                        (and
                          (exists (?t - cube_block ?s - chair)
                            (exists (?h - hexagonal_bin ?z ?e - (either triangle_block))
                              (exists (?v - building)
                                (exists (?j - hexagonal_bin)
                                  (forall (?u - game_object)
                                    (game-optional
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
                    (game-optional
                      (and
                        (agent_holds ?n)
                        (on desk)
                        (not
                          (not
                            (not
                              (not
                                (and
                                  (agent_holds ?m)
                                  (agent_holds ?m)
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
          (agent_holds ?q)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?w - (either golfball blue_cube_block) ?y - dodgeball)
      (and
        (preference preferenceA
          (exists (?l - cube_block ?i - game_object)
            (exists (?o - hexagonal_bin ?j - ball ?o - hexagonal_bin)
              (then
                (once (or (and (adjacent ?i) (rug_color_under ?o agent) ) (and (in ?i ?y) (< 7 (distance 3 ?y)) ) ) )
                (once (not (and (not (in_motion ?i) ) (not (not (and (agent_holds front) (on ?i) (and (between ?i ?i) (and (and (not (and (and (in ?y) (in_motion ?o) ) (in pink) (in_motion ?y ?o) (and (not (same_color ?y) ) ) ) ) (in_motion ?o) ) (adjacent ?i) ) (in_motion ?i) ) ) ) ) ) ) )
                (hold-while (agent_holds ?i ?y) (in_motion ?y) (and (> (distance_side agent ?o) 1) (in_motion ?o ?y) (and (not (agent_holds ?y) ) (not (< (distance ?o ?y) 1) ) ) ) )
                (hold-while (on ?i ?o) (and (in_motion ?i ?y) (in_motion ?o ?o) ) )
              )
            )
          )
        )
      )
    )
    (forall (?v - triangular_ramp)
      (and
        (preference preferenceB
          (exists (?n - hexagonal_bin ?d - hexagonal_bin)
            (at-end
              (agent_holds ?v)
            )
          )
        )
        (preference preferenceC
          (then
            (hold-while (in_motion agent ?v) (adjacent ?v) (and (in_motion ?v) (object_orientation ?v ?v) ) (in ?v ?v) )
            (hold (not (not (in desk ?v) ) ) )
            (hold (in ?v) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 3 180 )
    (< 18 (count preferenceA:yellow_pyramid_block) )
  )
)
(:scoring
  (+ (+ 10 (count preferenceC:dodgeball) )
    (* 10 (count preferenceB:wall:basketball:pink) )
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?e - red_dodgeball)
    (exists (?q - shelf)
      (and
        (and
          (exists (?p - dodgeball)
            (exists (?d - block ?o - color)
              (and
                (game-conserved
                  (< (distance ?p) (x_position 7 9))
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
    (forall (?v - block)
      (and
        (preference preferenceA
          (exists (?e - (either cube_block yellow_cube_block))
            (exists (?d - shelf ?f ?t ?s ?k ?c ?j - (either tall_cylindrical_block dodgeball dodgeball) ?w - curved_wooden_ramp)
              (then
                (once (exists (?u - (either cube_block yellow_cube_block)) (and (exists (?x - (either dodgeball yellow_cube_block)) (and (on ?v ?x) (and (agent_holds ?e) (agent_holds desk left) (not (on ?x) ) (or (not (adjacent ?u) ) (exists (?c - hexagonal_bin) (not (same_color ?c ?u) ) ) ) ) ) ) (in_motion ?e ?e) ) ) )
                (once (<= (distance ?v ?v) (distance bed)) )
                (once (agent_holds ?v) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (count-once-per-objects preferenceA:hexagonal_bin) (count-once preferenceA:basketball:hexagonal_bin:pink) 10 7 (count-once-per-objects preferenceA:red) (count preferenceA:beachball) (+ (>= (count-once-per-objects preferenceA:pink) 2 )
      )
      (count preferenceA)
      (+ (count preferenceA:dodgeball) (- (count-same-positions preferenceA:pink_dodgeball) )
      )
    )
    (* (count preferenceA:green:red) 2 )
  )
)
(:scoring
  (* (* (count preferenceA) 8 )
    3
  )
)
)


(define (game game-id) (:domain domain-name)
(:setup
  (exists (?n - watch ?a - chair ?q - (either blue_cube_block basketball top_drawer))
    (game-conserved
      (in_motion ?q)
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
  (count preferenceA:green)
)
)

