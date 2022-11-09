(define (game game-id-0) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (and
        (rug_color_under ?xxx)
        (not
          (in_motion ?xxx bed)
        )
      )
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?q - hexagonal_bin ?q - beachball)
      (and
        (preference preference1
          (at-end
            (and
              (touch ?q)
              (and
                (faces ?q)
                (in_motion ?q)
              )
              (in_motion ?q)
              (< (distance ?q ?q) (distance ?q 7))
              (agent_holds ?q)
              (on ?q ?q)
            )
          )
        )
        (preference preference2
          (then
            (once (not (and (not (and (agent_holds ?q) (agent_holds ?q) (and (not (not (touch ?q ?q) ) ) (exists (?a - ball ?w - shelf) (and (touch pink ?q) (not (and (in_motion ?w) (not (and (on ?q ?q) (and (exists (?n - triangular_ramp) (agent_holds ?q) ) (rug_color_under ?q) (< (distance front ?w) 1) (exists (?p - dodgeball) (agent_holds ?q agent) ) ) ) ) ) ) ) ) ) (agent_holds ?q ?q) ) ) (or (and (toggled_on ?q) (in ?q) ) (agent_holds ?q) ) ) ) )
            (once (> (distance room_center ?q) 2) )
            (once (between ?q) )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (in_motion ?xxx ?xxx) )
        (once (not (on ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= 5 (- 5 (total-score) ) )
)
(:scoring
  (count-once-per-objects preference2:book:dodgeball:hexagonal_bin)
)
)


(define (game game-id-1) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?n - (either ball yellow_cube_block) ?n - (either book cylindrical_block) ?b - ball)
      (exists (?g - block ?m ?o - (either basketball dodgeball bridge_block))
        (forall (?g - doggie_bed)
          (game-conserved
            (agent_holds ?o)
          )
        )
      )
    )
    (and
      (game-conserved
        (not
          (or
            (not
              (agent_holds ?xxx)
            )
            (in_motion agent ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?m - (either blue_cube_block tall_cylindrical_block golfball))
      (and
        (preference preference1
          (then
            (hold (equal_z_position ?m ?m) )
            (once (in_motion ?m desk) )
            (once (and (in_motion ?m) (and (and (and (agent_holds ?m) (and (agent_holds ?m) (in_motion floor) ) ) (not (on ?m desk) ) ) (adjacent ?m) (not (in_motion ?m ?m) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (> 15 3 )
)
(:scoring
  3
)
)


(define (game game-id-2) (:domain few-objects-room-v1)
(:setup
  (exists (?e - block)
    (forall (?y - wall)
      (game-optional
        (= 2 0 8)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t ?b - dodgeball)
        (then
          (once (in_motion ?t ?t) )
          (once (not (exists (?x - block ?z - hexagonal_bin) (on ?t) ) ) )
          (hold (and (on agent) (touch ?t ?b) ) )
        )
      )
    )
    (preference preference2
      (exists (?y - game_object)
        (at-end
          (not
            (not
              (not
                (agent_holds ?y)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference2:pink:dodgeball) (+ (count-unique-positions preference2:pink_dodgeball:golfball) (= (* (count preference1:dodgeball:pink_dodgeball) (* (* (count-once-per-objects preference2:dodgeball) (count preference2:blue_pyramid_block) )
            (* 4 (* (count preference2:hexagonal_bin) (count-once-per-objects preference2:yellow) )
              3
              (* 10 (count preference1:dodgeball) )
              (* 1 10 )
              (count-measure preference2:green)
            )
            2
          )
        )
        (* 5 (* (count preference1:golfball) (* (count-once-per-external-objects preference2:dodgeball:yellow:green) 2 (total-score) )
          )
        )
      )
    )
  )
)
(:scoring
  30
)
)


(define (game game-id-3) (:domain many-objects-room-v1)
(:setup
  (exists (?u - game_object ?a - bridge_block)
    (and
      (and
        (and
          (forall (?f - game_object ?h - hexagonal_bin)
            (game-conserved
              (< 1 1)
            )
          )
          (forall (?k - hexagonal_bin)
            (and
              (game-conserved
                (in ?k)
              )
            )
          )
        )
        (forall (?o ?c - hexagonal_bin)
          (and
            (game-optional
              (and
                (not
                  (not
                    (and
                      (on ?o ?c)
                      (and
                        (not
                          (in_motion ?c)
                        )
                        (not
                          (and
                            (agent_holds ?o agent)
                            (and
                              (not
                                (and
                                  (adjacent desk)
                                  (not
                                    (in_motion ?c ?c)
                                  )
                                )
                              )
                              (< 1 1)
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (in_motion ?c)
              )
            )
            (and
              (forall (?b ?u ?k ?j ?g - yellow_cube_block ?m - red_dodgeball)
                (exists (?y - ball)
                  (game-conserved
                    (in_motion agent)
                  )
                )
              )
            )
            (game-conserved
              (in ?o ?a)
            )
          )
        )
        (exists (?e - hexagonal_bin ?p - wall ?k - hexagonal_bin)
          (exists (?i - rug ?w - dodgeball ?l - game_object ?w - (either lamp curved_wooden_ramp))
            (game-conserved
              (not
                (and
                  (in ?a)
                  (on ?a upside_down)
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
    (forall (?q - hexagonal_bin ?v - dodgeball ?e - cylindrical_block)
      (and
        (preference preference1
          (exists (?p - (either dodgeball golfball cube_block))
            (then
              (once (on ?e ?p) )
              (once (in_motion agent ?p ?p) )
              (once (agent_holds ?e) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?e - wall)
        (then
          (once (and (touch agent) (and (or (and (not (and (same_type ?e ?e) (and (exists (?q - red_dodgeball ?y - beachball ?q - (either key_chain)) (agent_holds rug ?q ?q) ) (adjacent ?e) ) (not (agent_holds top_shelf) ) ) ) (and (adjacent ?e ?e) (< 1 0.5) (in_motion front_left_corner) (and (touch ?e ?e) (adjacent ?e ?e) ) ) ) ) (and (in_motion south_west_corner) (broken rug ?e) ) (in_motion rug) ) ) )
          (once (and (in ?e agent) (and (and (in_motion ?e) (< 1 1) (and (on agent) (not (agent_holds ?e) ) ) (and (on ?e agent) (and (not (on bed ?e) ) (not (agent_holds ?e bed) ) (agent_holds ?e) (and (adjacent ?e ?e) (on agent ?e) ) ) ) ) (in rug) ) ) )
          (once (or (and (adjacent ?e ?e) (in_motion agent ?e) ) (in_motion ?e) ) )
        )
      )
    )
    (preference preference3
      (exists (?b - color ?n - dodgeball)
        (then
          (once-measure (and (and (and (in_motion bed) (and (in_motion agent ?n) (agent_holds ?n) ) (in_motion ?n ?n) (in_motion ?n ?n) (and (in ?n) (or (not (not (agent_holds ?n ?n) ) ) (in ?n) ) (agent_holds ?n ?n) ) (in ?n) ) (not (and (not (on ?n) ) (in ?n ?n) ) ) ) (in ?n top_drawer) ) (building_size 4 ?n) )
          (once (agent_holds ?n) )
          (hold-while (touch ?n ?n) (not (> 5 1) ) )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count-same-positions preference1:alarm_clock:dodgeball) (count preference3:doggie_bed:hexagonal_bin) )
    (>= 5 2 )
    (>= (count-once-per-objects preference2:dodgeball) (count-once-per-objects preference1:basketball) )
  )
)
(:scoring
  5
)
)


(define (game game-id-4) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (not
          (or
            (and
              (on ?xxx ?xxx)
              (adjacent ?xxx)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (touch ?xxx door) )
        (once (in ?xxx) )
        (once (and (in_motion ?xxx ?xxx) (< 3 (distance desk ?xxx)) ) )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:pink_dodgeball) (count preference1:top_drawer) )
    (>= (count preference1:dodgeball:red) 2 )
  )
)
(:scoring
  (count preference1:dodgeball)
)
)


(define (game game-id-5) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (faces ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?i - hexagonal_bin)
      (and
        (preference preference1
          (then
            (any)
            (once (touch desk ?i) )
            (once (not (> 1 (distance agent ?i)) ) )
          )
        )
      )
    )
    (preference preference2
      (exists (?b - curved_wooden_ramp ?z - cube_block)
        (exists (?j - hexagonal_bin)
          (then
            (hold (forall (?a - (either dodgeball)) (agent_holds ?j ?a) ) )
            (hold (and (and (and (not (not (in ?j) ) ) (and (not (agent_holds ?j ?j) ) (in ?z) ) ) (and (on tan) (exists (?x - hexagonal_bin) (not (exists (?a - shelf) (in_motion ?x) ) ) ) ) ) (not (agent_holds pink ?z) ) (and (and (and (on upright ?z) (and (same_type ?j agent) (on ?z brown) ) ) (= (distance ?j ?j) 1 (distance ?z 10)) (and (on ?j) (and (on ?j ?z) (agent_holds ?z) ) ) ) (not (agent_holds ?z) ) ) ) )
            (once (agent_holds ?z) )
          )
        )
      )
    )
    (forall (?r - dodgeball)
      (and
        (preference preference3
          (exists (?q - hexagonal_bin)
            (exists (?z - building)
              (at-end
                (not
                  (and
                    (not
                      (not
                        (not
                          (not
                            (and
                              (not
                                (toggled_on ?q)
                              )
                              (agent_holds ?q)
                              (in_motion ?r)
                            )
                          )
                        )
                      )
                    )
                    (< (distance ?q ?q) 1)
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
  (>= (- (count preference3:pink_dodgeball) )
    (count preference2:dodgeball)
  )
)
(:scoring
  50
)
)


(define (game game-id-6) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in ?xxx)
      (agent_holds ?xxx)
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - dodgeball ?w - hexagonal_bin)
        (then
          (hold (same_object ?w) )
          (hold (and (and (in ?w) (touch ?w) (not (on ?w) ) ) (exists (?k - teddy_bear) (adjacent ?k) ) ) )
          (once (in_motion ?w ?w) )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= 5 (count preference1) )
  )
)
(:scoring
  (count preference1:golfball:yellow)
)
)


(define (game game-id-7) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - pillow)
        (then
          (once (agent_holds ?e) )
          (once (> 1 (distance ?e)) )
          (hold (= 1) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:tan) (count preference1:yellow) )
    (= (+ (count-total preference1:dodgeball:golfball) (count-once-per-external-objects preference1:dodgeball) )
      5
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-8) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (> (distance ?xxx 9) 2)
  )
)
(:constraints
  (and
    (forall (?o - dodgeball)
      (and
        (preference preference1
          (exists (?r - watch)
            (exists (?m - cube_block)
              (forall (?d - block)
                (then
                  (once (and (agent_holds ?o) (not (equal_z_position ?d ?m) ) ) )
                  (once (= 1 1) )
                  (once (not (in_motion ?o bed) ) )
                )
              )
            )
          )
        )
        (preference preference2
          (then
            (once (on ?o ?o) )
            (hold (agent_holds bed) )
          )
        )
      )
    )
    (preference preference3
      (then
        (hold-while (and (agent_holds ?xxx) (in_motion ?xxx ?xxx) (touch ?xxx ?xxx) ) (not (< (distance 4 ?xxx) 7) ) )
        (once (agent_holds ?xxx) )
        (once (< (distance agent ?xxx) (distance ?xxx ?xxx)) )
      )
    )
    (preference preference4
      (then
        (once (same_color ?xxx) )
        (hold-while (not (> (distance 1 2) 7) ) (and (not (not (and (< 1 1) (agent_holds ?xxx ?xxx ?xxx) ) ) ) (and (not (and (and (on ?xxx) (not (on ?xxx) ) ) (not (= 1 (x_position ?xxx ?xxx)) ) ) ) (not (and (in_motion ?xxx) (or (agent_holds ?xxx) (and (agent_holds floor ?xxx) (agent_holds ?xxx sideways) ) ) ) ) ) ) (exists (?y - game_object) (< (distance room_center 4) (distance ?y desk)) ) (on ?xxx) )
        (once (and (and (adjacent ?xxx ?xxx) (not (in ?xxx ?xxx) ) ) (and (adjacent agent) (in_motion ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (< (count-once-per-objects preference4:red) 12 )
)
(:scoring
  40
)
)


(define (game game-id-9) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (exists (?h - ball)
        (exists (?u - (either cube_block hexagonal_bin) ?j - beachball ?g - hexagonal_bin)
          (game-conserved
            (adjacent ?g green_golfball)
          )
        )
      )
    )
    (exists (?n - rug)
      (game-conserved
        (and
          (and
            (and
              (in_motion ?n ?n)
              (touch ?n)
            )
            (agent_holds ?n)
          )
          (in_motion ?n ?n)
          (on ?n)
        )
      )
    )
    (exists (?q - doggie_bed ?q - (either yellow_cube_block wall) ?f - hexagonal_bin)
      (forall (?s - dodgeball)
        (game-optional
          (and
            (and
              (in_motion ?f ?f)
              (< (distance ?s ?f) 2)
            )
            (in ?s)
            (< 0 (distance ?s ?f))
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - (either pyramid_block cube_block))
        (at-end
          (is_setup_object bed ?k)
        )
      )
    )
    (preference preference2
      (then
        (once (and (not (agent_holds ?xxx) ) (and (object_orientation ?xxx bed) (agent_holds ?xxx desk) ) ) )
        (once (on rug) )
        (once (and (not (broken ?xxx) ) (in_motion ?xxx) ) )
      )
    )
  )
)
(:terminal
  (> 2 (count preference1:dodgeball) )
)
(:scoring
  (* (count-once-per-objects preference2:dodgeball) (count-total preference2:purple) )
)
)


(define (game game-id-10) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (>= (distance desk door ?xxx) 0.5)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - building)
        (exists (?c - block ?j - golfball)
          (exists (?b - game_object)
            (exists (?s - doggie_bed)
              (exists (?g - triangular_ramp ?r - game_object)
                (exists (?z ?w - teddy_bear ?t - (either yellow_cube_block laptop) ?i - hexagonal_bin ?q - (either tall_cylindrical_block hexagonal_bin pyramid_block))
                  (exists (?l - hexagonal_bin)
                    (then
                      (once (in ?k) )
                      (hold-while (not (in_motion ?r pink) ) (touch ?q) (and (opposite ?r) ) )
                      (hold-for 3 (in_motion bed rug) )
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
  (or
    (> (count preference1:basketball) (not (count-once-per-objects preference1:pink:red) ) )
    (not
      (>= (count preference1:blue_pyramid_block:wall) (count-same-positions preference1:blue_dodgeball) )
    )
  )
)
(:scoring
  (count-once-per-external-objects preference1:yellow)
)
)


(define (game game-id-11) (:domain medium-objects-room-v1)
(:setup
  (not
    (game-optional
      (agent_holds bed)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (exists (?g - doggie_bed) (not (and (and (and (and (adjacent ?g) (in_motion ?g) ) (in ?g agent) ) (agent_holds agent ?g) ) (not (< (distance side_table) 6) ) ) ) ) ) )
        (once (in_motion door) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 15 (count preference1:cube_block) )
)
(:scoring
  (count preference1:pyramid_block)
)
)


(define (game game-id-12) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-conserved
          (in ?xxx ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (in_motion ?xxx ?xxx) (and (same_color ?xxx) (exists (?j - hexagonal_bin) (and (not (adjacent_side ?j ?j) ) (in_motion ?j agent) ) ) ) )
        (once (in_motion ?xxx ?xxx) )
        (once (same_type ?xxx desk) )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= (>= (count-same-positions preference1:basketball:brown) (* 4 (* (= (count preference1:dodgeball) (not (count preference1:red:pink) ) )
              (* (external-forall-maximize (* 5 3 )
                )
                (* (+ (* 10 (or (count preference1:green) (count preference1:dodgeball:basketball) ) )
                    (count preference1:dodgeball:rug)
                  )
                  4
                )
              )
              25
            )
          )
        )
        (* 10 16 )
      )
    )
    (or
      (or
        (>= (+ (count preference1:purple:dodgeball) (count preference1:beachball) )
          (+ (* (+ (/ (external-forall-maximize (count-measure preference1:yellow) ) (* (* (total-score) 0 )
                    (* (* (count-once-per-objects preference1:yellow) 2 )
                      3
                    )
                  )
                )
                (count preference1:pink)
              )
              (+ 9 (* (= (or 8 ) (total-time) )
                  (count-once-per-objects preference1:pink)
                  2
                )
              )
            )
            (count-once-per-objects preference1:red)
          )
        )
        (or
          (or
            (and
              (or
                (>= (count preference1:beachball:dodgeball) (count-once-per-objects preference1:dodgeball) )
                (>= 9 (+ (* (* (count preference1:dodgeball) 5 )
                      (- (* (count preference1:dodgeball) (count-measure preference1:green:dodgeball:yellow) )
                      )
                    )
                    (count-once-per-objects preference1:dodgeball)
                  )
                )
              )
            )
            (>= (count preference1:dodgeball) (count-once-per-objects preference1:pink) )
          )
          (or
            (>= (count-once-per-objects preference1:basketball:pink) 5 )
            (or
              (>= (count preference1:pink_dodgeball:yellow) 2 )
            )
          )
        )
        (>= 3 (* (count preference1:cylindrical_block:hexagonal_bin) 2 )
        )
      )
      (>= (count preference1:pink) 5 )
      (> 7 (+ (- (count-once-per-external-objects preference1:beachball) )
          (count preference1:rug:dodgeball)
        )
      )
    )
    (or
      (<= (count preference1:orange:hexagonal_bin) (count-once-per-objects preference1:dodgeball) )
      (< 10 (external-forall-minimize (count preference1:orange) ) )
    )
  )
)
(:scoring
  (count preference1:red:yellow)
)
)


(define (game game-id-13) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (and
        (in_motion pink)
        (or
          (not
            (and
              (not
                (< 10 6)
              )
              (exists (?z ?q - game_object)
                (agent_holds ?q ?q)
              )
            )
          )
          (agent_holds ?xxx)
        )
      )
      (not
        (exists (?r - chair ?l - hexagonal_bin ?y - ball)
          (and
            (on ?y ?y)
            (agent_holds ?y ?y)
            (agent_holds ?y)
            (and
              (and
                (adjacent ?y desk)
                (and
                  (agent_holds ?y)
                  (adjacent ?y)
                )
                (and
                  (and
                    (agent_holds ?y ?y)
                    (or
                      (not
                        (on sideways ?y)
                      )
                      (in ?y)
                    )
                  )
                  (= 9 2 7)
                )
              )
              (not
                (not
                  (agent_holds ?y)
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
    (preference preference1
      (exists (?i ?x ?e ?s - hexagonal_bin)
        (then
          (once (in_motion ?e ?s) )
          (once (agent_holds ?x) )
          (hold (adjacent ?e) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 2 )
)
(:scoring
  (count-once-per-objects preference1:yellow)
)
)


(define (game game-id-14) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (not
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - dodgeball ?n - (either block blue_cube_block))
        (at-end
          (in ?n)
        )
      )
    )
    (preference preference2
      (exists (?f - hexagonal_bin)
        (then
          (hold-to-end (touch ?f) )
          (hold (not (rug_color_under ?f ?f) ) )
          (hold-while (not (and (not (and (not (and (agent_holds ?f) (not (agent_holds ?f) ) ) ) (adjacent floor) ) ) (on ?f) ) ) (and (and (same_object ?f) (touch ?f) ) (not (agent_holds ?f) ) ) )
        )
      )
    )
    (forall (?r - pillow)
      (and
        (preference preference3
          (then
            (once (and (in_motion ?r) (and (in_motion ?r ?r) (agent_holds ?r ?r) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (= (and 2 (* (count preference3:golfball:dodgeball) (count preference3:pink) )
      (-
        2
        60
      )
    )
    (+ (+ (external-forall-maximize 2 ) 4 )
      (* 40 (count preference2:blue_dodgeball:hexagonal_bin) )
    )
  )
)
(:scoring
  (count-once-per-objects preference3:yellow:basketball:red)
)
)


(define (game game-id-15) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?y - (either cd dodgeball))
      (and
        (exists (?b - hexagonal_bin)
          (forall (?g - block ?a - (either blue_cube_block pyramid_block))
            (game-conserved
              (in_motion ?a)
            )
          )
        )
      )
    )
    (game-conserved
      (and
        (in_motion ?xxx)
        (agent_holds ?xxx)
      )
    )
    (not
      (game-conserved
        (on ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (and (and (and (not (adjacent agent) ) ) (not (and (in_motion ?xxx ?xxx) (and (not (and (in_motion ?xxx) ) ) (> (distance ?xxx ?xxx) (distance 7 green_golfball)) ) ) ) ) (touch blue ?xxx) (or (in upright) (not (not (agent_holds ?xxx) ) ) ) ) (in_motion ?xxx) (and (agent_holds ?xxx) (in_motion ?xxx ?xxx ?xxx) ) )
        (once (not (and (in_motion rug) ) ) )
        (once (agent_holds ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
    (forall (?t - hexagonal_bin)
      (and
        (preference preference2
          (exists (?f - (either book wall))
            (at-end
              (in_motion ?f)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (<= (* (count-once-per-objects preference2:book) (count-measure preference2:pink:basketball) )
      (count-once-per-objects preference2:red)
    )
    (>= (count preference1:brown:golfball) (count-unique-positions preference2:pink:basketball:basketball) )
  )
)
(:scoring
  (* (count preference1:basketball) (count-once preference1:beachball) )
)
)


(define (game game-id-16) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (adjacent ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - hexagonal_bin)
        (then
          (once (adjacent_side side_table) )
          (once (not (agent_holds ?f) ) )
          (forall-sequence (?q - cube_block)
            (then
              (once (and (agent_holds ?f ?f) (and (in_motion ?q ?q) ) ) )
              (hold-while (not (in_motion agent) ) (adjacent ?q) (on agent ?q) (and (< 1 1) (not (>= 1 1) ) ) )
              (once (on ?f ?q) )
            )
          )
        )
      )
    )
    (forall (?f - cube_block)
      (and
        (preference preference2
          (exists (?s - (either pyramid_block dodgeball beachball))
            (at-end
              (in_motion bed ?s)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count preference1:beachball) (count preference2:blue_pyramid_block) )
    (or
      (and
        (>= (count-once-per-objects preference2:hexagonal_bin) 50 )
        (>= (external-forall-minimize (* (count preference1:green:pink) (count-once-per-objects preference2) )
          )
          (-
            (count-once-per-objects preference1:green)
            (* 2 (count-once-per-objects preference1:beachball:yellow) )
          )
        )
      )
      (>= 15 (+ (count preference1:blue_dodgeball) )
      )
      (>= (count preference2:dodgeball:side_table:basketball) (count-once-per-objects preference2:beachball:basketball) )
    )
  )
)
(:scoring
  (* (= (+ (count preference2:purple) 15 )
      (count-measure preference2:yellow)
    )
    (count preference1:yellow_pyramid_block:book)
  )
)
)


(define (game game-id-17) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (equal_z_position ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - game_object ?h - ball)
        (then
          (hold (and (< 4 1) (adjacent ?h) ) )
          (hold-while (in_motion ?h) (not (and (in agent agent) ) ) )
          (once (in_motion bed ?h) )
          (once (in_motion agent ?h) )
        )
      )
    )
    (preference preference2
      (exists (?o ?j ?x ?i ?k ?l - (either yellow_cube_block bridge_block pyramid_block) ?l - dodgeball ?s - drawer)
        (exists (?i - game_object)
          (then
            (hold-to-end (agent_holds ?i ?s) )
            (once (on ?i) )
            (hold-while (not (not (in_motion ?s) ) ) (agent_holds ?s ?s) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (or
        (>= 100 (count-once-per-objects preference2:purple) )
        (>= (count-once preference1:dodgeball) (* (* 4 (count preference1:block) )
            (* (count preference2:pink) (total-score) )
          )
        )
      )
    )
    (>= (count-once-per-objects preference1:hexagonal_bin) (count preference2:basketball) )
  )
)
(:scoring
  (* (count-once preference1:pink) (* (- (count-once preference2:wall) )
      (+ (count preference2:golfball) 2 )
    )
  )
)
)


(define (game game-id-18) (:domain medium-objects-room-v1)
(:setup
  (exists (?k - pyramid_block ?c - game_object)
    (game-optional
      (in_motion agent ?c)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (agent_holds ?xxx) (or (not (adjacent ?xxx ?xxx) ) (and (in_motion ?xxx) (not (agent_holds back) ) ) ) ) )
        (once (on ?xxx ?xxx) )
        (forall-sequence (?e - building)
          (then
            (once (not (adjacent pink_dodgeball) ) )
            (once-measure (and (between ?e ?e) ) (distance desk ?e) )
            (once (in_motion ?e) )
          )
        )
      )
    )
  )
)
(:terminal
  (< (* (count-once-per-objects preference1:golfball:basketball) 3 )
    (count preference1:cube_block:golfball)
  )
)
(:scoring
  3
)
)


(define (game game-id-19) (:domain few-objects-room-v1)
(:setup
  (forall (?b - beachball)
    (game-conserved
      (agent_holds ?b ?b)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx) )
        (hold-while (touch ?xxx) (and (not (on green_golfball) ) (on ?xxx ?xxx) ) (and (in_motion ?xxx) (and (in ?xxx ?xxx) (and (between ?xxx ?xxx) (in_motion ?xxx) ) ) ) )
        (once (and (not (agent_holds ?xxx) ) (in ?xxx) (is_setup_object ?xxx) ) )
      )
    )
  )
)
(:terminal
  (not
    (>= 5 1 )
  )
)
(:scoring
  (external-forall-maximize
    (count preference1:basketball:purple)
  )
)
)


(define (game game-id-20) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (in_motion ?xxx)
      )
    )
    (preference preference2
      (exists (?q - hexagonal_bin)
        (then
          (once (> 1 (distance )) )
          (once (adjacent ?q) )
          (once (agent_holds ?q) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:purple) (count-once-per-objects preference2:orange:hexagonal_bin) )
)
(:scoring
  (and
    9
  )
)
)


(define (game game-id-21) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?y - cube_block)
      (or
        (game-conserved
          (and
            (not
              (on ?y ?y ?y)
            )
            (and
              (not
                (agent_holds ?y)
              )
              (agent_holds ?y ?y)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - chair ?l - (either main_light_switch))
        (then
          (any)
          (hold (on floor ?l) )
          (once (in_motion agent desk) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (count-overlapping preference1:dodgeball) (- (count-shortest preference1) )
    )
    (>= (count-once-per-objects preference1:dodgeball) 9 )
  )
)
(:scoring
  7
)
)


(define (game game-id-22) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (exists (?d - game_object)
        (in_motion ?d ?d ?d)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - beachball ?f - dodgeball)
        (exists (?j - building)
          (exists (?a - curved_wooden_ramp ?e - desk_shelf)
            (exists (?a - cube_block)
              (then
                (once (same_object ?e) )
                (once (agent_holds ?a) )
                (once (< 1 1) )
                (once (and (adjacent ?e) (on ?e ?f) (not (not (in_motion ?a) ) ) (or (not (and (on ?e) (and (game_over ?a ?j) (on ?a brown) ) (agent_holds ?f) (and (and (and (agent_holds front) (agent_holds ?f) (above ?a) ) (agent_holds ?f ?a) (and (not (or (< (distance front_left_corner ?f) 7) (agent_holds desk ?f) ) ) (or (and (forall (?h - curved_wooden_ramp) (not (in ?j ?e) ) ) (touch ?a ?j) ) (adjacent ?a ?f) ) ) ) (not (agent_holds floor ?a) ) ) (= (distance ) (distance ?a ?a) (distance ?f ?f)) (in ?a) (in upright floor ?f ?f) (faces ?j) ) ) (not (same_object agent) ) ) ) )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (at-end
        (on ?xxx ?xxx)
      )
    )
    (forall (?p - dodgeball)
      (and
        (preference preference3
          (at-end
            (in_motion ?p agent)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (count preference3:dodgeball) )
)
(:scoring
  (= (count preference2:golfball) (count preference3:brown) )
)
)


(define (game game-id-23) (:domain medium-objects-room-v1)
(:setup
  (exists (?x - (either key_chain laptop) ?n - tall_cylindrical_block ?o - color)
    (not
      (and
        (game-optional
          (< (distance ?o) 0)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?z - dodgeball)
      (and
        (preference preference1
          (exists (?g - (either book blue_cube_block))
            (then
              (once (on ?z ?z) )
              (hold (not (in_motion ?z) ) )
              (once (not (in_motion ?g ?z) ) )
            )
          )
        )
      )
    )
    (forall (?d - dodgeball)
      (and
        (preference preference2
          (at-end
            (agent_holds ?d)
          )
        )
      )
    )
    (preference preference3
      (then
        (once (same_color ?xxx agent) )
        (hold-while (not (agent_holds ?xxx) ) (in ?xxx ?xxx) (exists (?y - (either cube_block)) (agent_holds ?y) ) )
        (once (not (not (not (and (not (touch ?xxx ?xxx) ) (same_color ?xxx) ) ) ) ) )
      )
    )
  )
)
(:terminal
  (> 3 (* (* 3 (count-once-per-external-objects preference1:top_drawer) (+ (count preference2:pink) 5 )
        7
        (external-forall-maximize
          (* (count-once preference3:blue_cube_block) (+ (external-forall-maximize (count-once-per-objects preference2:pink) ) (* 2 (count preference2:red) )
            )
            (count-once preference3:block)
          )
        )
        (count preference2:dodgeball)
      )
      (count preference1:pink)
    )
  )
)
(:scoring
  300
)
)


(define (game game-id-24) (:domain few-objects-room-v1)
(:setup
  (forall (?p - hexagonal_bin)
    (forall (?w - hexagonal_bin)
      (exists (?e - watch)
        (or
          (forall (?f - cube_block ?v - dodgeball)
            (forall (?s - hexagonal_bin)
              (and
                (game-conserved
                  (agent_holds ?p)
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
    (preference preference1
      (exists (?g - doggie_bed)
        (exists (?y - dodgeball)
          (exists (?l - (either cellphone dodgeball) ?s - (either cylindrical_block flat_block pyramid_block))
            (exists (?x - (either dodgeball book) ?c - (either cd pyramid_block doggie_bed))
              (exists (?v - wall)
                (exists (?k - cube_block ?w - dodgeball)
                  (exists (?b - hexagonal_bin ?j - chair)
                    (exists (?p ?r - cube_block ?k - (either laptop pillow blue_cube_block) ?x - dodgeball ?u - wall)
                      (at-end
                        (in_motion ?y)
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
    (preference preference2
      (then
        (hold (agent_holds ?xxx) )
      )
    )
    (preference preference3
      (then
        (once (agent_holds ?xxx) )
        (once (agent_holds ?xxx) )
        (once (not (in_motion ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (count preference3:pink) (* (/ 0 50 ) (or (external-forall-maximize (+ (total-time) (total-score) )
        )
        3
        (* (count-once-per-objects preference2:tan) (count preference2:yellow_cube_block) 2 )
      )
    )
  )
)
(:scoring
  (* (+ (* (total-time) (* (count preference2:blue_pyramid_block) (* (* (count-once preference1:beachball) (* (* (count preference2:yellow) (count preference3:yellow:hexagonal_bin) )
                (count preference3:book)
              )
            )
            0
          )
        )
      )
      300
      (count preference3:hexagonal_bin)
    )
    (count preference2:dodgeball)
  )
)
)


(define (game game-id-25) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (>= (distance ?xxx ?xxx) 6)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - dodgeball)
        (then
          (once (and (not (and (not (not (adjacent ?e) ) ) (adjacent pillow) ) ) (same_type ?e) ) )
          (once (forall (?v - color) (exists (?o - curved_wooden_ramp) (agent_holds ?v) ) ) )
          (once (adjacent ?e desk) )
        )
      )
    )
  )
)
(:terminal
  (<= 12 (= 10 (+ (* (* (* 10 10 )
            (count-once-per-objects preference1:pink)
          )
          (count-same-positions preference1:book)
          (* (count preference1:yellow) (count-measure preference1:dodgeball) )
          (count-once-per-external-objects preference1:basketball)
          (not
            (total-score)
          )
          5
        )
        (count-longest preference1:dodgeball:pink_dodgeball)
      )
      20
    )
  )
)
(:scoring
  (* 6 3 )
)
)


(define (game game-id-26) (:domain medium-objects-room-v1)
(:setup
  (forall (?y - hexagonal_bin)
    (and
      (game-optional
        (in_motion agent)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (touch ?xxx) )
        (once (not (touch ?xxx ?xxx) ) )
        (once-measure (adjacent ?xxx) (distance 10 10) )
      )
    )
  )
)
(:terminal
  (and
    (or
      (>= 10 (count preference1:pink) )
      (>= (* (* 15 3 (count preference1:pink_dodgeball) )
          (count preference1:dodgeball)
        )
        (total-time)
      )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-27) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (game-optional
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (forall (?m ?i ?b ?f - game_object ?x - doggie_bed ?p - game_object)
        (then
          (once (not (on rug) ) )
          (once (agent_holds sideways) )
          (hold-while (and (agent_holds ?p) (on ?p ?p) (exists (?m ?a ?r ?c - (either flat_block mug alarm_clock)) (on ?p ?a) ) ) (and (on ?p ?p) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count-overlapping preference1:cube_block) (= (* (* (count preference1:dodgeball) (count-once-per-objects preference1:pink) )
          5
        )
        15
      )
      (count-once preference1:hexagonal_bin:cube_block)
      (external-forall-maximize
        (* (count preference1:orange) 8 )
      )
      6
      4
    )
    (count preference1:basketball)
  )
)
(:scoring
  (* 2 5 )
)
)


(define (game game-id-28) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (and
      (not
        (and
          (agent_holds ?xxx)
          (not
            (in_motion ?xxx ?xxx)
          )
        )
      )
      (agent_holds ?xxx)
      (on desk)
    )
  )
)
(:constraints
  (and
    (forall (?k - drawer ?x - hexagonal_bin ?i - building)
      (and
        (preference preference1
          (then
            (hold (agent_holds agent ?i) )
            (hold (in_motion ?i) )
            (hold (in ?i) )
          )
        )
        (preference preference2
          (then
            (once (game_over ?i) )
            (hold (object_orientation ?i) )
            (once (not (not (not (and (and (not (and (in_motion ?i) (not (in_motion desk ?i) ) (exists (?s - curved_wooden_ramp) (and (and (and (exists (?z - hexagonal_bin) (agent_holds ?z pink_dodgeball) ) (in_motion bed) ) ) (and (and (> 5 1) (and (and (adjacent ?s ?i) (not (on ?s) ) ) (not (agent_holds ?s) ) (on ?s) ) (in ?s ?i) ) (in_motion ?s ?i) ) ) ) (agent_holds ?i) (and (agent_holds ?i) (not (and (agent_holds pink desk) (in ?i ?i) ) ) (not (and (in_motion ?i) ) ) ) (in ?i ?i) (agent_holds ?i ?i) ) ) (in_motion door) ) (not (in_motion agent ?i) ) ) ) ) ) )
          )
        )
      )
    )
    (preference preference3
      (exists (?c - (either laptop pen cellphone))
        (exists (?u - hexagonal_bin ?w - game_object ?o - pyramid_block)
          (then
            (once (>= (distance ?c ?o) (distance ?c side_table)) )
            (once (adjacent ?c) )
            (hold (adjacent bed) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:dodgeball) (* (- 2 )
      5
      15
    )
  )
)
(:scoring
  (count-once-per-objects preference1:side_table)
)
)


(define (game game-id-29) (:domain medium-objects-room-v1)
(:setup
  (forall (?j - shelf)
    (exists (?m - dodgeball)
      (game-conserved
        (and
          (and
            (not
              (touch ?m)
            )
            (and
              (on ?j)
              (in_motion ?j)
            )
            (< (distance agent ?j) 1)
          )
          (in_motion ?j)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?y - dodgeball)
      (and
        (preference preference1
          (then
            (hold (on ?y bed) )
            (once (exists (?q - hexagonal_bin) (in_motion ?q) ) )
            (forall-sequence (?t - game_object)
              (then
                (once (and (>= (distance 7 desk) (distance ?y ?y)) (on ?y) ) )
                (once-measure (in_motion agent) (distance back room_center) )
                (hold (not (on ?t ?y) ) )
              )
            )
            (once (on ?y) )
          )
        )
        (preference preference2
          (then
            (once (agent_holds ?y ?y) )
            (hold (not (not (not (and (in ?y ?y) (and (and (adjacent pink_dodgeball ?y) (not (not (adjacent pink) ) ) ) (game_start ?y) (in_motion ?y floor) (agent_holds ?y) ) ) ) ) ) )
            (once (in_motion ?y ?y) )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count preference2:blue_cube_block) (count-shortest preference1:dodgeball) )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-30) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?d - ball ?g - teddy_bear)
      (game-conserved
        (exists (?v - hexagonal_bin)
          (touch ?g ?g ?v)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?w - game_object)
      (and
        (preference preference1
          (exists (?x - wall)
            (then
              (once (in_motion ?w ?w) )
              (once (in_motion ?w ?x) )
              (once (not (agent_holds top_shelf ?w) ) )
            )
          )
        )
      )
    )
    (forall (?y - (either game_object lamp cellphone) ?v - building)
      (and
        (preference preference2
          (exists (?y - dodgeball ?k - ball)
            (exists (?z - (either blue_cube_block credit_card laptop) ?u - game_object)
              (exists (?i - dodgeball)
                (exists (?s - beachball ?x - ball ?r - cube_block)
                  (then
                    (once (not (in_motion ?i) ) )
                    (hold (agent_holds ?u) )
                    (once (agent_holds ?k ?r) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?r - hexagonal_bin)
        (then
          (hold (or (not (in ?r) ) (above ?r ?r) (exists (?x - (either dodgeball desktop)) (in ?r) ) (in_motion ?r ?r) ) )
          (once (adjacent agent ?r) )
          (once (is_setup_object ?r) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-total preference1:pyramid_block) (* (+ 3 (* (* (+ (count-once-per-objects preference1:blue) (count preference1:hexagonal_bin) )
            (count-once-per-external-objects preference2:basketball)
          )
          (+ (count-once preference2:white) (* (count-once-per-external-objects preference3:yellow_cube_block) 10 (count preference2:blue_dodgeball) )
          )
          (count preference2:basketball:dodgeball:dodgeball)
        )
        (count-once preference1:book:blue:beachball)
        (* 20 (count preference2:golfball) )
      )
      (count-once-per-objects preference2:green)
    )
  )
)
(:scoring
  (* 50 (external-forall-maximize (count-once-per-objects preference2:basketball) ) )
)
)


(define (game game-id-31) (:domain few-objects-room-v1)
(:setup
  (forall (?j - game_object)
    (exists (?x - desk_shelf)
      (game-optional
        (in_motion ?x ?x)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (not (agent_holds ?xxx door) ) )
        (once (not (on agent ?xxx) ) )
        (once (and (on ?xxx) (not (agent_holds brown) ) ) )
      )
    )
  )
)
(:terminal
  (>= 5 4 )
)
(:scoring
  300
)
)


(define (game game-id-32) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?v - (either tall_cylindrical_block yellow_cube_block) ?k - dodgeball)
      (game-conserved
        (on rug)
      )
    )
    (game-conserved
      (and
        (not
          (in_motion ?xxx ?xxx)
        )
        (and
          (not
            (agent_holds ?xxx ?xxx)
          )
          (faces ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?e - block)
      (and
        (preference preference1
          (exists (?q - hexagonal_bin ?p - cube_block)
            (exists (?r - teddy_bear)
              (forall (?m - dodgeball)
                (exists (?o - wall)
                  (exists (?f - triangular_ramp)
                    (then
                      (once (agent_holds ?e) )
                      (once (on pillow) )
                      (once (on ?m ?m) )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
    (forall (?a - hexagonal_bin)
      (and
        (preference preference2
          (then
            (once (> (distance ?a ?a) 0) )
            (hold (in ?a) )
            (hold-while (exists (?b - building) (adjacent ?b) ) (agent_holds ?a ?a) )
          )
        )
        (preference preference3
          (at-end
            (> (distance ?a 9) 2)
          )
        )
        (preference preference4
          (exists (?c - chair)
            (exists (?p - ball)
              (forall (?u - hexagonal_bin ?g - doggie_bed)
                (exists (?i - chair)
                  (then
                    (once (agent_holds ?i ?g) )
                    (once (exists (?q - chair ?n - block) (in_motion ?n) ) )
                    (hold (and (touch ?c) (< 9 2) (or (agent_holds ?p ?i) (and (< (distance ?i ?a) (distance_side ?a ?a)) (agent_holds ?a) ) ) (in_motion ?a) (adjacent ?i) (agent_holds ?i ?i) (agent_holds ?i ?c) (not (< (distance ?g ?c) 1) ) (and (not (and (is_setup_object ?a) (touch agent) ) ) (agent_holds ?i) ) (and (and (and (touch ?a) (agent_holds pink) ) (in_motion ?g) ) (not (and (in pillow ?a) (agent_holds ?a ?c) (agent_holds ?i) ) ) (in_motion ?g) ) (in_motion ?i ?g) (agent_holds ?c sideways desk) ) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference5
      (then
        (once (on ?xxx) )
        (once (in_motion ?xxx) )
        (once (in_motion desk) )
      )
    )
  )
)
(:terminal
  (and
    (or
      (= 3 40 )
    )
    (not
      (or
        (>= (external-forall-maximize (count preference5:dodgeball) ) (count-once-per-objects preference4:basketball) )
        (>= 1 (* 10 (count preference4:dodgeball:yellow:pink) )
        )
      )
    )
  )
)
(:scoring
  (* (* (count-once-per-objects preference5:dodgeball) (count-once-per-external-objects preference5:beachball) )
    (* 6 (count-once-per-objects preference1:beachball:pink) )
    3
  )
)
)


(define (game game-id-33) (:domain medium-objects-room-v1)
(:setup
  (forall (?n - game_object ?c - chair ?n - block)
    (game-optional
      (adjacent_side ?n ?n)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (on desk ?xxx) )
        (forall-sequence (?l - dodgeball)
          (then
            (hold (not (agent_holds bed ?l) ) )
            (hold (agent_holds ?l) )
            (hold-while (exists (?u - hexagonal_bin) (in ?u) ) (in_motion ?l) )
          )
        )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (> (external-forall-maximize (count preference1:triangle_block) ) (count-once-per-objects preference1:pyramid_block) )
)
(:scoring
  (count-overlapping preference1:beachball:pink)
)
)


(define (game game-id-34) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-conserved
          (and
            (agent_holds agent ?xxx)
            (in_motion ?xxx)
            (in_motion ?xxx)
          )
        )
        (exists (?n - (either flat_block desktop))
          (exists (?y - game_object)
            (game-conserved
              (and
                (not
                  (equal_z_position ?y ?y)
                )
                (in_motion ?n)
              )
            )
          )
        )
        (game-optional
          (< (distance room_center ?xxx) 1)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - (either golfball dodgeball))
        (exists (?v ?k - chair)
          (then
            (any)
            (once (agent_holds ?a ?k) )
            (once (not (in_motion ?v ?v) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count-once-per-objects preference1:basketball) (count-once-per-objects preference1:blue_cube_block) )
    (> 4 5 )
    (or
      (and
        (< 20 1 )
        (>= (count-once-per-objects preference1:golfball) 10 )
        (<= 12 10 )
      )
      (and
        (>= 2 (count-unique-positions preference1:pink) )
        (or
          (and
            (<= 10 (count preference1:doggie_bed:beachball) )
            (>= (total-time) (count-same-positions preference1) )
          )
          (>= (* 2 (+ 10 (* 3 (count preference1:orange) )
              )
            )
            (<= 2 (/ (* (count preference1:dodgeball) (count-shortest preference1:rug) )
                (external-forall-maximize
                  (count-once preference1:beachball)
                )
              )
            )
          )
          (or
            (and
              (> (count-measure preference1:triangle_block:basketball) (count-once-per-objects preference1:basketball:alarm_clock:beachball) )
            )
          )
        )
      )
    )
  )
)
(:scoring
  (count-once preference1:pink_dodgeball)
)
)


(define (game game-id-35) (:domain few-objects-room-v1)
(:setup
  (forall (?x - cube_block)
    (game-conserved
      (in ?x)
    )
  )
)
(:constraints
  (and
    (forall (?f - hexagonal_bin)
      (and
        (preference preference1
          (then
            (once (in ?f) )
            (hold (in ?f ?f) )
            (once-measure (in_motion ?f ?f) (distance ?f front_left_corner) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (not (count-once-per-external-objects preference1:blue_cube_block) ) (count preference1:dodgeball) )
)
(:scoring
  (* (count-once preference1:doggie_bed) (count-once-per-objects preference1:red:pink_dodgeball) )
)
)


(define (game game-id-36) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - hexagonal_bin)
        (then
          (once-measure (object_orientation ?w) (distance 4 ?w ?w) )
          (once (touch ?w ?w) )
          (hold-while (in_motion bed agent) (not (not (equal_z_position ?w) ) ) (and (not (and (in ?w ?w) (agent_holds ?w) ) ) (rug_color_under ?w ?w) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (+ 15 15 )
      5
    )
  )
)
(:scoring
  6
)
)


(define (game game-id-37) (:domain many-objects-room-v1)
(:setup
  (exists (?x - hexagonal_bin)
    (game-conserved
      (not
        (on desktop front)
      )
    )
  )
)
(:constraints
  (and
    (forall (?z - (either cd teddy_bear blue_cube_block))
      (and
        (preference preference1
          (forall (?t - ball)
            (exists (?a - hexagonal_bin)
              (exists (?e - cube_block)
                (then
                  (hold (and (in_motion ?a) (agent_holds ?a) ) )
                  (once (and (and (on ?a) (in_motion ?e) ) (not (on ?e) ) ) )
                  (once (and (agent_holds ?a ?a) (on ?t ?t) ) )
                )
              )
            )
          )
        )
        (preference preference2
          (exists (?l - ball ?h - dodgeball)
            (then
              (once (in ?h) )
              (once (agent_holds ?h ?h pink) )
              (once (touch ?z) )
            )
          )
        )
      )
    )
    (preference preference3
      (forall (?l - hexagonal_bin ?o - dodgeball)
        (exists (?n - dodgeball)
          (then
            (hold (and (and (agent_holds ?o) (agent_holds ?n) ) (and (not (same_color ?n ?o) ) (or (in_motion ?o ?o) (in_motion ?n) ) ) ) )
            (once (not (in_motion ?o ?n) ) )
            (hold (and (on ?n) (in_motion ?n) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (* (+ (and (* 180 (>= (total-time) 40 )
                (external-forall-maximize
                  2
                )
                1
                (count-once-per-external-objects preference3)
                2
                (count-once-per-objects preference3:doggie_bed)
                (count-unique-positions preference1:beachball)
                (count preference3:golfball)
              )
              (external-forall-maximize
                2
              )
            )
            4
            5
            5
          )
          (> 10 300 )
        )
        30
      )
      10
    )
  )
)
(:scoring
  (count-measure preference3:yellow)
)
)


(define (game game-id-38) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?b - block)
      (game-conserved
        (on ?b)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (and
          (open blue ?xxx)
          (in_motion ?xxx ?xxx ?xxx)
        )
      )
    )
  )
)
(:terminal
  (>= 3 (external-forall-maximize (* (count preference1:golfball) (external-forall-maximize (count-same-positions preference1:dodgeball) ) )
    )
  )
)
(:scoring
  6
)
)


(define (game game-id-39) (:domain many-objects-room-v1)
(:setup
  (or
    (forall (?y - doggie_bed ?g - (either pencil pencil) ?c - cube_block)
      (game-conserved
        (agent_holds brown)
      )
    )
  )
)
(:constraints
  (and
    (forall (?j - dodgeball ?s - cube_block)
      (and
        (preference preference1
          (then
            (once (not (not (and (game_start ?s) (in_motion ?s ?s) ) ) ) )
            (hold-for 6 (equal_z_position ?s) )
            (once (in_motion left ?s) )
          )
        )
        (preference preference2
          (exists (?a - (either floor dodgeball))
            (exists (?t - dodgeball ?c - dodgeball)
              (at-end
                (in block ?a)
              )
            )
          )
        )
        (preference preference3
          (exists (?p - dodgeball ?n ?y ?t - hexagonal_bin ?b - dodgeball)
            (then
              (once (on ?s) )
              (hold-while (in_motion ?s agent) (agent_holds ?b) )
            )
          )
        )
      )
    )
    (forall (?t - doggie_bed)
      (and
        (preference preference4
          (exists (?l - hexagonal_bin)
            (then
              (hold (in ?l) )
              (once (not (and (adjacent ?l ?l ?l) (in_motion ?l) (< 2 1) ) ) )
              (any)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 7 (count-once-per-objects preference2:purple:golfball:pink) )
)
(:scoring
  (total-score)
)
)


(define (game game-id-40) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (and
      (forall (?o - (either doggie_bed blue_cube_block))
        (in_motion ?o)
      )
      (agent_holds agent)
    )
  )
)
(:constraints
  (and
    (forall (?p - hexagonal_bin)
      (and
        (preference preference1
          (then
            (hold (adjacent ?p ?p) )
            (once (and (agent_holds green ?p) (< 1 9) ) )
            (once (same_color ?p ?p) )
            (hold (not (and (in_motion ?p) (not (not (adjacent rug) ) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (and (count-once preference1:hexagonal_bin) ) 20 )
)
(:scoring
  2
)
)


(define (game game-id-41) (:domain few-objects-room-v1)
(:setup
  (forall (?m - doggie_bed)
    (game-conserved
      (on ?m)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (and (agent_holds ?xxx ?xxx) (and (not (touch ?xxx) ) (in ?xxx) ) ) )
        (once (in_motion ?xxx) )
        (once (in_motion ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count preference1:pink_dodgeball) (count-once preference1:beachball) )
)
(:scoring
  (* 15 2 (* 8 (count preference1:golfball) 0.7 )
    (+ 5 (+ (* (count preference1:dodgeball:yellow) (* (* 10 2 )
            (count-once preference1:dodgeball)
          )
        )
        (* (* (count-once preference1:doggie_bed) )
          (count preference1:blue_cube_block:book)
        )
      )
    )
    1
    (* (count-once-per-objects preference1:yellow:purple) (* (count preference1:pink) (* 2 (- (external-forall-minimize (count preference1:golfball) ) )
        )
      )
      (count-once preference1:blue_dodgeball:golfball)
    )
  )
)
)


(define (game game-id-42) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (forall (?w - ball ?i - shelf)
      (and
        (preference preference1
          (at-end
            (and
              (not
                (in ?i)
              )
              (agent_holds ?i)
            )
          )
        )
        (preference preference2
          (exists (?m - hexagonal_bin)
            (exists (?n - block)
              (exists (?s - hexagonal_bin)
                (exists (?b - dodgeball)
                  (exists (?p - ball ?u - block ?a - hexagonal_bin ?u - chair)
                    (exists (?c - game_object ?x - (either top_drawer cube_block bridge_block))
                      (then
                        (once (agent_holds ?s ?n) )
                        (once (not (on green_golfball) ) )
                        (hold (not (and (in ?i) (not (not (not (in_motion ?m ?n) ) ) ) ) ) )
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
  (or
    (>= (<= 20 (count-once-per-objects preference2:green:red_pyramid_block) )
      5
    )
  )
)
(:scoring
  (* (external-forall-minimize 3 ) (count preference1:yellow_cube_block) )
)
)


(define (game game-id-43) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - teddy_bear ?l - (either dodgeball golfball))
        (then
          (once (on ?l ?l) )
          (once (and (and (in_motion ?l) (touch ?l) ) (and (rug_color_under ?l ?l) (agent_holds ) ) ) )
          (once (in_motion rug ?l) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:golfball) 1 )
)
(:scoring
  (-
    (- (count-once-per-objects preference1:blue_dodgeball) )
    (+ (count preference1:dodgeball:yellow) (* 8 (total-time) )
      (total-time)
      (count preference1:basketball)
    )
  )
)
)


(define (game game-id-44) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (< (distance 6) (distance ?xxx 0))
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (agent_holds ?xxx ?xxx) )
        (once (on ?xxx brown) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
    (preference preference2
      (then
        (once (not (agent_holds ?xxx) ) )
        (once (agent_holds desk) )
        (once (in_motion ?xxx) )
      )
    )
  )
)
(:terminal
  (not
    (>= 15 (* 3 (count-once preference1:purple) )
    )
  )
)
(:scoring
  (count preference2:dodgeball)
)
)


(define (game game-id-45) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - wall)
        (exists (?l - (either chair triangular_ramp))
          (then
            (hold (on ?u) )
            (once (and (and (in_motion desk) (same_color ?u) ) (and (in ?u ?u) (on ?l ?l) (not (not (not (not (in_motion ?u ?u) ) ) ) ) (in_motion ?u) ) ) )
            (once (agent_holds ?l ?u) )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (and (agent_holds ?xxx ?xxx) ) )
        (once (on ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (and
    (= (* 2 (count-total preference1:wall:bed) )
      3
    )
  )
)
(:scoring
  (count-once-per-objects preference2:dodgeball:dodgeball)
)
)


(define (game game-id-46) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (not
          (in ?xxx)
        )
      )
      (and
        (game-conserved
          (or
            (agent_holds ?xxx ?xxx)
            (toggled_on brown)
          )
        )
        (exists (?f - ball)
          (game-conserved
            (in ?f)
          )
        )
      )
      (and
        (and
          (game-optional
            (and
              (same_color ?xxx ?xxx)
              (in_motion ?xxx yellow)
            )
          )
        )
        (and
          (forall (?e - hexagonal_bin ?e - wall)
            (game-conserved
              (not
                (< (distance ?e 9) (distance agent ?e))
              )
            )
          )
          (and
            (exists (?j - dodgeball)
              (game-conserved
                (on ?j ?j)
              )
            )
          )
          (forall (?j - ball)
            (and
              (game-optional
                (agent_holds ?j ?j)
              )
            )
          )
          (forall (?w - dodgeball)
            (exists (?a - block)
              (and
                (and
                  (forall (?p - hexagonal_bin ?v - dodgeball)
                    (game-conserved
                      (in_motion upright)
                    )
                  )
                  (and
                    (and
                      (game-conserved
                        (agent_holds bed ?a)
                      )
                      (game-conserved
                        (agent_holds ?w ?a)
                      )
                    )
                  )
                )
              )
            )
          )
          (exists (?v - flat_block ?b - (either main_light_switch cube_block) ?n - curved_wooden_ramp)
            (forall (?c - (either dodgeball bridge_block) ?h - pyramid_block)
              (game-conserved
                (not
                  (not
                    (exists (?q - hexagonal_bin)
                      (not
                        (same_color ?n)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (forall (?g ?t ?d ?w ?z ?x - hexagonal_bin ?d - hexagonal_bin)
          (exists (?r - ball)
            (and
              (forall (?s - (either pillow book))
                (and
                  (forall (?t - (either key_chain yellow))
                    (game-optional
                      (on ?r ?s)
                    )
                  )
                )
              )
              (and
                (game-optional
                  (touch ?r ?r)
                )
                (not
                  (and
                    (game-conserved
                      (and
                        (and
                          (not
                            (agent_holds ?r ?d)
                          )
                          (adjacent ?d)
                          (not
                            (same_color ?d)
                          )
                          (not
                            (in_motion ?r)
                          )
                          (and
                            (not
                              (not
                                (not
                                  (same_color ?d)
                                )
                              )
                            )
                            (agent_holds ?d)
                          )
                          (in_motion ?d)
                        )
                        (same_color ?d)
                      )
                    )
                    (exists (?o - wall ?k - triangular_ramp ?j - (either bridge_block doggie_bed))
                      (and
                        (game-optional
                          (agent_holds ?r ?d)
                        )
                        (forall (?c - chair)
                          (game-conserved
                            (game_start ?d)
                          )
                        )
                      )
                    )
                    (game-conserved
                      (in_motion ?d ?r)
                    )
                  )
                )
                (game-conserved
                  (not
                    (in_motion bridge_block ?d)
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
        (in_motion ?xxx)
      )
      (exists (?t - game_object)
        (and
          (forall (?u - hexagonal_bin ?g - block ?c - dodgeball)
            (and
              (exists (?j - dodgeball)
                (game-conserved
                  (object_orientation desk agent)
                )
              )
              (and
                (forall (?a - hexagonal_bin)
                  (forall (?f - (either triangular_ramp yellow_cube_block))
                    (not
                      (and
                        (game-conserved
                          (on ?a)
                        )
                        (and
                          (game-conserved
                            (touch ?t)
                          )
                          (forall (?j - hexagonal_bin)
                            (game-conserved
                              (agent_holds ?a)
                            )
                          )
                          (exists (?h - dodgeball)
                            (or
                              (game-conserved
                                (not
                                  (touch ?c)
                                )
                              )
                              (game-optional
                                (and
                                  (and
                                    (in_motion ?f ?c)
                                    (not
                                      (agent_holds ?c)
                                    )
                                  )
                                  (object_orientation bridge_block ?t)
                                  (and
                                    (forall (?j - block)
                                      (on ?f ?a)
                                    )
                                    (not
                                      (in ?t rug)
                                    )
                                  )
                                )
                              )
                              (game-optional
                                (agent_holds ?t ?a)
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (exists (?a - rug ?d - ball)
                  (forall (?i - hexagonal_bin ?r ?x ?n - dodgeball)
                    (exists (?j - game_object ?p - cube_block ?k - golfball ?v - hexagonal_bin)
                      (and
                        (and
                          (game-conserved
                            (on ?r)
                          )
                        )
                        (game-conserved
                          (in upright ?t)
                        )
                        (and
                          (forall (?y - ball)
                            (game-optional
                              (not
                                (agent_holds desk ?y)
                              )
                            )
                          )
                          (and
                            (forall (?k - game_object ?z - hexagonal_bin)
                              (forall (?f ?l - (either alarm_clock pyramid_block))
                                (forall (?h - ball ?e - ball ?u - (either chair yellow_cube_block dodgeball))
                                  (and
                                    (and
                                      (forall (?y - hexagonal_bin)
                                        (and
                                          (game-optional
                                            (and
                                              (and
                                                (not
                                                  (agent_holds ?u)
                                                )
                                                (and
                                                  (touch ?v ?d)
                                                  (touch ?u)
                                                  (in_motion bed ?l)
                                                )
                                              )
                                              (< (distance agent desk) 1)
                                            )
                                          )
                                          (and
                                            (forall (?q - ball ?j - triangular_ramp ?o - triangular_ramp)
                                              (exists (?m - dodgeball)
                                                (game-conserved
                                                  (touch ?r ?c)
                                                )
                                              )
                                            )
                                          )
                                          (game-conserved
                                            (agent_holds ?n ?f)
                                          )
                                        )
                                      )
                                    )
                                  )
                                )
                              )
                            )
                            (and
                              (forall (?w - hexagonal_bin)
                                (forall (?f - building)
                                  (forall (?j - color ?l ?i ?p - game_object)
                                    (or
                                      (game-conserved
                                        (agent_holds ?n)
                                      )
                                      (and
                                        (exists (?j - doggie_bed)
                                          (and
                                            (and
                                              (game-optional
                                                (in_motion ?j agent)
                                              )
                                              (or
                                                (and
                                                  (game-conserved
                                                    (in ?v)
                                                  )
                                                  (exists (?u - cube_block)
                                                    (exists (?o - (either golfball wall))
                                                      (game-conserved
                                                        (in_motion agent)
                                                      )
                                                    )
                                                  )
                                                  (and
                                                    (and
                                                      (exists (?e - blinds ?s ?g - yellow_cube_block)
                                                        (forall (?u - desktop ?b - wall)
                                                          (or
                                                            (game-conserved
                                                              (and
                                                                (on ?w door)
                                                                (touch ?s agent)
                                                              )
                                                            )
                                                            (forall (?a - wall)
                                                              (game-conserved
                                                                (in_motion ?a ?w)
                                                              )
                                                            )
                                                            (game-conserved
                                                              (in ?w)
                                                            )
                                                          )
                                                        )
                                                      )
                                                    )
                                                    (and
                                                      (game-optional
                                                        (on ?r ?p)
                                                      )
                                                    )
                                                    (exists (?z - dodgeball)
                                                      (game-conserved
                                                        (on ?l)
                                                      )
                                                    )
                                                    (game-optional
                                                      (agent_holds ?p ?r)
                                                    )
                                                    (game-conserved
                                                      (not
                                                        (in_motion ?f)
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
                                (agent_holds ?r)
                              )
                              (and
                                (and
                                  (and
                                    (game-optional
                                      (in_motion ?c ?r)
                                    )
                                    (game-optional
                                      (agent_holds ?r)
                                    )
                                    (game-optional
                                      (not
                                        (and
                                          (in ?c ?t)
                                          (and
                                            (in_motion ?t)
                                            (agent_holds ?t)
                                          )
                                        )
                                      )
                                    )
                                  )
                                )
                                (game-conserved
                                  (touch ?n)
                                )
                              )
                            )
                          )
                          (forall (?f ?h ?y ?e - red_dodgeball ?e - ball)
                            (and
                              (game-conserved
                                (agent_holds ?e)
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (forall (?f - ball ?l - flat_block)
                  (game-optional
                    (in_motion ?l ?c)
                  )
                )
              )
              (and
                (exists (?s - hexagonal_bin)
                  (game-optional
                    (agent_holds ?c ?t)
                  )
                )
                (game-conserved
                  (and
                    (not
                      (in ?t ?t)
                    )
                    (touch ?c agent)
                  )
                )
              )
            )
          )
        )
      )
      (forall (?j - hexagonal_bin ?n - teddy_bear)
        (exists (?y - doggie_bed)
          (exists (?b ?i ?c - hexagonal_bin)
            (exists (?a - chair)
              (exists (?u - pillow)
                (and
                  (game-conserved
                    (agent_holds ?c ?y)
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
    (forall (?m - wall)
      (and
        (preference preference1
          (exists (?t - dodgeball ?a - building)
            (exists (?o - teddy_bear)
              (exists (?c - dodgeball)
                (at-end
                  (and
                    (and
                      (adjacent bed)
                      (exists (?r - ball)
                        (in_motion ?a)
                      )
                    )
                    (and
                      (< (distance room_center ?o) 1)
                      (and
                        (and
                          (agent_holds agent ?m)
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
  )
)
(:terminal
  (>= 5 (external-forall-maximize (* 10 (count preference1:hexagonal_bin) 100 3 10 3 )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-47) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (touch ?xxx) )
        (hold (and (and (not (not (agent_holds ?xxx) ) ) (not (not (on ?xxx) ) ) ) (not (in ?xxx) ) ) )
        (once (agent_holds desk ?xxx) )
      )
    )
    (preference preference2
      (exists (?e - dodgeball ?x - dodgeball)
        (then
          (once (agent_holds ?x) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (* (count-total preference2:yellow) 10 )
  )
)
(:scoring
  (* 300 (count preference2:basketball) 10 6 )
)
)


(define (game game-id-48) (:domain medium-objects-room-v1)
(:setup
  (exists (?t - hexagonal_bin ?f ?x - dodgeball)
    (exists (?z - ball)
      (game-conserved
        (adjacent ?z)
      )
    )
  )
)
(:constraints
  (and
    (forall (?z - block ?a - hexagonal_bin ?u - teddy_bear ?w - (either laptop pink chair dodgeball bridge_block pyramid_block desktop))
      (and
        (preference preference1
          (then
            (hold (in_motion ?w ?w ?w) )
            (hold (and (in_motion pillow) (on ?w) ) )
            (once (and (agent_holds ?w) (in agent) (on ?w) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (> (> (* (count preference1:basketball) 4 )
      (* 1 (count preference1:pyramid_block:pink) )
    )
    5
  )
)
(:scoring
  (+ (count preference1:red_pyramid_block) (count preference1:dodgeball) )
)
)


(define (game game-id-49) (:domain medium-objects-room-v1)
(:setup
  (or
    (forall (?u - flat_block)
      (exists (?l - teddy_bear ?l - sliding_door)
        (exists (?j - doggie_bed)
          (exists (?k - color ?s ?k - building)
            (game-optional
              (and
                (not
                  (not
                    (in_motion ?u)
                  )
                )
                (<= (distance ?l ?j) (distance ?u ?k))
              )
            )
          )
        )
      )
    )
    (game-optional
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - hexagonal_bin)
        (exists (?a - yellow_cube_block ?t - hexagonal_bin)
          (then
            (once (in_motion front) )
            (once (and (in_motion ?n) ) )
            (hold (not (exists (?h - block ?v - doggie_bed ?o - hexagonal_bin) (in_motion front ?o) ) ) )
          )
        )
      )
    )
    (forall (?p - ball ?n - (either watch cube_block hexagonal_bin))
      (and
        (preference preference2
          (exists (?q - (either chair side_table) ?g - curved_wooden_ramp)
            (then
              (hold (exists (?o ?f - dodgeball ?a - doggie_bed) (in ?g ?a) ) )
              (once-measure (object_orientation ) (distance ?n ?g) )
              (once (not (not (in_motion ?g) ) ) )
              (once (adjacent desk ?g) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 7 5 )
)
(:scoring
  (count-once-per-objects preference2:purple)
)
)


(define (game game-id-50) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?f ?z ?d ?s - chair)
      (forall (?o - dodgeball)
        (and
          (exists (?r - (either bridge_block dodgeball triangle_block yellow_cube_block))
            (game-optional
              (and
                (on ?f)
                (not
                  (in ?s ?o)
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
    (preference preference1
      (then
        (once (agent_holds agent) )
        (once (not (on desk ?xxx) ) )
        (hold (touch ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (<= (/ (+ (- 10 )
        (count preference1:blue_dodgeball)
      )
      (count preference1)
    )
    (+ (+ (* (* (external-forall-maximize (count-overlapping preference1:yellow) ) (count preference1:red_pyramid_block:green) )
          (= (* 10 10 )
            (* (* 2 (and (count preference1:yellow) 5 2 ) )
              1
            )
          )
        )
        (+ (count-once preference1:hexagonal_bin) (count preference1:basketball) )
      )
      1
    )
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id-51) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (adjacent ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - ball)
        (exists (?d - curved_wooden_ramp ?m - hexagonal_bin)
          (exists (?l ?t - hexagonal_bin)
            (exists (?c - ball)
              (exists (?z - (either cylindrical_block dodgeball) ?g - cube_block)
                (at-end
                  (and
                    (not
                      (not
                        (not
                          (in ?o ?m)
                        )
                      )
                    )
                    (not
                      (and
                        (in_motion bed)
                        (not
                          (exists (?a - dodgeball)
                            (not
                              (not
                                (and
                                  (not
                                    (agent_holds ?a)
                                  )
                                  (in ?g)
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
    )
    (preference preference2
      (exists (?y ?g ?r - hexagonal_bin)
        (then
          (hold (in_motion ?r) )
          (hold-while (agent_holds desk) (in_motion ?g) )
          (hold (in ?r) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:beachball) (count preference2:golfball:top_drawer) )
)
(:scoring
  10
)
)


(define (game game-id-52) (:domain few-objects-room-v1)
(:setup
  (and
    (game-optional
      (not
        (adjacent ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - building)
        (forall (?c - building ?r - game_object ?w - (either beachball pyramid_block dodgeball pink laptop tall_cylindrical_block lamp))
          (exists (?r - triangular_ramp)
            (at-end
              (and
                (same_object ?s ?r)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (- 1 )
    7
  )
)
(:scoring
  8
)
)


(define (game game-id-53) (:domain many-objects-room-v1)
(:setup
  (exists (?s - dodgeball)
    (game-optional
      (>= 1 5)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (agent_holds agent ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (>= 2 (count-once-per-objects preference1:orange) )
    (>= (count-once-per-objects preference1:dodgeball:bed) (< (+ 10 (count preference1:book) )
        (- (total-time) )
      )
    )
    (>= (count preference1:basketball) 2 )
  )
)
(:scoring
  2
)
)


(define (game game-id-54) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (adjacent ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (in door ?xxx) (or (in_motion agent ?xxx) (agent_holds ?xxx) ) ) )
        (hold (not (adjacent rug) ) )
        (once (< 1 (distance room_center ?xxx)) )
      )
    )
    (preference preference2
      (at-end
        (opposite ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (>= (* (count-once-per-objects preference2:pink_dodgeball:yellow) 5 )
      (count preference2:blue_cube_block)
    )
    (>= (count preference1:dodgeball) (total-time) )
  )
)
(:scoring
  1
)
)


(define (game game-id-55) (:domain medium-objects-room-v1)
(:setup
  (forall (?e - curved_wooden_ramp)
    (game-conserved
      (between ?e)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (adjacent ?xxx ?xxx) )
        (once (not (agent_holds ?xxx ?xxx front) ) )
        (once (not (agent_holds ?xxx) ) )
      )
    )
    (preference preference2
      (then
        (once (agent_holds ?xxx) )
        (once (exists (?p - (either bridge_block golfball) ?z - dodgeball) (in ?z ?z) ) )
      )
    )
  )
)
(:terminal
  (< (* (count-once-per-objects preference2:red) (external-forall-maximize (* 1 (count preference2:pink) )
      )
    )
    (count preference1:red_pyramid_block:basketball)
  )
)
(:scoring
  (count preference1:golfball)
)
)


(define (game game-id-56) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (in_motion rug ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (in ?xxx) )
        (once (and (not (on ?xxx) ) (is_setup_object ?xxx) ) )
        (hold (in_motion ?xxx agent) )
        (once (same_object desk floor) )
        (once (in_motion bed ?xxx ?xxx) )
      )
    )
    (preference preference2
      (then
        (once (and (in_motion ?xxx) (agent_holds ?xxx) ) )
        (once (in_motion ?xxx) )
        (once (or (in ?xxx) (in ?xxx bed) (and (not (and (agent_holds ?xxx) (agent_holds ?xxx ?xxx) ) ) (in_motion pink_dodgeball ?xxx) ) ) )
        (once (game_over ?xxx) )
      )
    )
    (forall (?s ?c ?m ?p ?k ?v - shelf)
      (and
        (preference preference3
          (then
            (once (on ?k ?p) )
            (once (not (in ?s) ) )
            (hold (in agent ?p) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 2 (count-once-per-objects preference3:yellow:pink) )
)
(:scoring
  3
)
)


(define (game game-id-57) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (and
        (in_motion ?xxx ?xxx)
        (agent_holds ?xxx)
      )
      (and
        (agent_holds agent)
        (= 3 10)
      )
    )
  )
)
(:constraints
  (and
    (forall (?e ?y ?j - block)
      (and
        (preference preference1
          (then
            (once (not (in_motion ?e) ) )
            (hold-while (and (and (adjacent ?y) (not (and (in_motion ?y ?y) (on ?j) (agent_holds ?j) ) ) (and (game_start ?y) (touch ?y) (or (in ?j) (not (not (not (touch floor) ) ) ) ) ) (agent_holds agent) ) (agent_holds ?j) (agent_holds agent) ) (same_color ?e) )
            (hold-while (on pink_dodgeball ?j) (< 1 10) (on pink_dodgeball ?y) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:pink) (+ (* (external-forall-minimize (total-time) ) (count-once-per-objects preference1:beachball) )
      (count preference1:orange:blue_dodgeball)
      (- 15 )
    )
  )
)
(:scoring
  (* (* (count-once-per-objects preference1:beachball:hexagonal_bin) (+ 9 (* 4 (count preference1:hexagonal_bin:beachball) )
      )
    )
    (* (count preference1:beachball:dodgeball) (count-once-per-objects preference1:beachball) (count preference1) 10 (- (+ (count preference1:top_drawer) (= (external-forall-minimize (* (* 10 (- (count preference1:blue_dodgeball:pink_dodgeball) )
                )
                (external-forall-maximize
                  (count-once-per-objects preference1:yellow)
                )
              )
            )
            (<= 1 6 )
          )
          10
        )
      )
      (* 50 10 )
    )
  )
)
)


(define (game game-id-58) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (on agent ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx ?xxx) )
        (once (agent_holds ?xxx) )
        (once (not (and (same_color ?xxx block) (and (touch ?xxx pink_dodgeball) (and (in_motion ?xxx ?xxx) (not (agent_holds ?xxx) ) ) ) (in_motion ?xxx ?xxx) ) ) )
      )
    )
    (preference preference2
      (exists (?p - (either cd blue_cube_block alarm_clock))
        (then
          (once (in_motion ?p) )
          (once (object_orientation ?p) )
          (once (in_motion bed) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:blue_pyramid_block) (count-once-per-objects preference2:rug) )
)
(:scoring
  (count preference1:top_drawer)
)
)


(define (game game-id-59) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (or
      (not
        (agent_holds ?xxx pink)
      )
      (agent_holds ?xxx ?xxx ?xxx ?xxx)
      (and
        (<= 10 1)
        (in_motion ?xxx)
        (in_motion agent)
      )
      (adjacent ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (same_color ?xxx)
      )
    )
  )
)
(:terminal
  (> (count-once-per-objects preference1:white:blue_cube_block) (count preference1:golfball) )
)
(:scoring
  (count-once-per-objects preference1:pink)
)
)


(define (game game-id-60) (:domain medium-objects-room-v1)
(:setup
  (forall (?t ?z - dodgeball)
    (and
      (and
        (and
          (game-conserved
            (not
              (in ?t)
            )
          )
          (game-conserved
            (not
              (not
                (adjacent ?t ?z)
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
    (forall (?b ?v ?h ?t - book ?l - block)
      (and
        (preference preference1
          (then
            (hold (and (not (rug_color_under ?l) ) (forall (?c - (either pen doggie_bed)) (adjacent ?c ?l) ) ) )
            (once (in ?l) )
            (hold (agent_holds ?l ?l) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:golfball) 2 )
)
(:scoring
  (count-shortest preference1:doggie_bed)
)
)


(define (game game-id-61) (:domain medium-objects-room-v1)
(:setup
  (forall (?p - shelf)
    (game-conserved
      (between ?p ?p)
    )
  )
)
(:constraints
  (and
    (forall (?c - hexagonal_bin ?e - book)
      (and
        (preference preference1
          (then
            (once (and (on ?e ?e) (and (not (in_motion south_wall) ) (in_motion agent ?e) ) ) )
            (once (forall (?w - drawer) (exists (?j - pyramid_block) (agent_holds ?j) ) ) )
            (hold-while (in_motion ?e agent) (not (not (agent_holds ?e ?e) ) ) )
          )
        )
        (preference preference2
          (exists (?m - hexagonal_bin)
            (exists (?t - wall ?l - doggie_bed)
              (at-end
                (agent_holds ?l)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (* 2 3 )
      6
    )
    (count preference1:pink)
  )
)
(:scoring
  2
)
)


(define (game game-id-62) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?o - teddy_bear ?g - watch)
      (exists (?b - (either game_object game_object alarm_clock) ?f - triangular_ramp ?c - game_object ?o - hexagonal_bin)
        (game-conserved
          (exists (?u - cube_block ?t - cube_block)
            (and
              (and
                (exists (?a - cube_block)
                  (in_motion ?a)
                )
                (in_motion ?t ?t)
              )
              (agent_holds ?t ?o)
              (touch ?o)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (in_motion ?xxx) ) )
        (hold (in ?xxx) )
        (hold-while (in_motion ?xxx bed) (adjacent ?xxx ?xxx) (in_motion ?xxx ?xxx) )
        (once (and (forall (?c - pillow ?a - building) (in_motion ?a) ) (in_motion ?xxx ?xxx) ) )
      )
    )
    (preference preference2
      (exists (?u - ball)
        (at-end
          (adjacent ?u)
        )
      )
    )
  )
)
(:terminal
  (or
    (> (count preference2:golfball) (* 1 (< (+ (count-measure preference1:golfball) (* (* (count preference1:doggie_bed) (count-once-per-objects preference1:basketball) )
              (+ (count preference2:brown) (count preference1:blue_cube_block) 3 5 (* (count preference2:side_table:tan) (< (count preference2:beachball) (count preference1:pink) )
                )
                (count preference2:blue_pyramid_block:orange)
              )
            )
          )
          5
        )
      )
    )
    (>= (- 20 )
      3
    )
    (and
      (or
        (>= (external-forall-maximize (count-once-per-objects preference1:beachball) ) (count-once preference1:pink:dodgeball) )
        (> 2 (total-score) )
      )
    )
  )
)
(:scoring
  (count-once-per-objects preference1:white)
)
)


(define (game game-id-63) (:domain medium-objects-room-v1)
(:setup
  (exists (?h - color)
    (and
      (exists (?u - hexagonal_bin)
        (game-optional
          (and
            (and
              (in_motion agent)
              (not
                (touch front bed)
              )
              (in ?h ?u)
            )
            (in ?u)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (and
          (not
            (not
              (exists (?b - building)
                (agent_holds ?b front)
              )
            )
          )
          (agent_holds pink)
          (in ?xxx ?xxx)
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 0 (total-score) )
    (> (count-once-per-objects preference1:dodgeball) (count-once-per-external-objects preference1:basketball) )
  )
)
(:scoring
  (* 3 2 )
)
)


(define (game game-id-64) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in ?xxx ?xxx)
      (adjacent rug)
      (adjacent_side ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (not (in_motion ?xxx) ) ) )
        (once (or (rug_color_under ?xxx) (in_motion ?xxx) (> (distance agent) 1) (in ?xxx) ) )
        (once (in_motion ?xxx front) )
      )
    )
  )
)
(:terminal
  (> 6 (count preference1:golfball:pink) )
)
(:scoring
  6
)
)


(define (game game-id-65) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (and
      (agent_holds ?xxx pink_dodgeball)
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (in_motion ?xxx rug) )
        (once (and (not (not (and (is_setup_object top_shelf ?xxx) (adjacent ?xxx) ) ) ) (not (and (and (agent_holds ?xxx) (< 1 (distance ?xxx door)) ) (in_motion ?xxx) (and (on ?xxx) (adjacent ?xxx ?xxx) (not (and (not (agent_holds ?xxx) ) (agent_holds bed) ) ) (equal_z_position bed) ) ) ) ) )
        (once (in_motion pink_dodgeball) )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= 3 100 )
    )
    (> (* (* 1 (count preference1:hexagonal_bin) (= (total-score) (count-unique-positions preference1:dodgeball) )
          (+ (- 5 )
            5
            (* (* (and 5 4 30 ) 10 (count-once-per-external-objects preference1:golfball) )
              (+ 3 5 )
            )
          )
        )
        1
        2
      )
      (* (count preference1:basketball:book:hexagonal_bin) (- (count preference1:golfball) )
      )
    )
  )
)
(:scoring
  (* (count preference1:beachball) (count preference1:pink:dodgeball) (count-once-per-objects preference1) (count-once-per-external-objects preference1:tall_cylindrical_block) (* 2 (count-once preference1:dodgeball) )
    10
  )
)
)


(define (game game-id-66) (:domain medium-objects-room-v1)
(:setup
  (exists (?u - hexagonal_bin)
    (and
      (forall (?k - ball)
        (and
          (and
            (and
              (forall (?n - block)
                (or
                  (game-optional
                    (and
                      (agent_holds ?k)
                      (exists (?q - dodgeball)
                        (not
                          (on agent ?k)
                        )
                      )
                      (adjacent ?n)
                    )
                  )
                )
              )
              (and
                (game-conserved
                  (and
                    (on ?k ?k)
                    (not
                      (agent_holds ?u ?k)
                    )
                    (not
                      (not
                        (not
                          (game_over ?u ?k)
                        )
                      )
                    )
                  )
                )
                (and
                  (game-conserved
                    (agent_holds ?u)
                  )
                )
              )
            )
          )
        )
      )
      (game-conserved
        (or
          (not
            (in_motion ?u)
          )
          (on ?u)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - hexagonal_bin)
        (at-end
          (same_type ?a ?a)
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (not (total-time) ) (* 7 10 (count-once-per-objects preference1:beachball:red) (count-once-per-objects preference1:basketball:green) )
    )
    (<= (count-once-per-objects preference1:dodgeball) (total-score) )
  )
)
(:scoring
  (- (count preference1:golfball) )
)
)


(define (game game-id-67) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?y - hexagonal_bin ?s - chair)
      (game-optional
        (in_motion ?s)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx) )
        (hold (and (and (agent_holds ?xxx) (adjacent_side ?xxx) ) (agent_holds ?xxx) ) )
        (once (not (and (and (in_motion yellow) (not (< (distance 4 ?xxx ?xxx) 6) ) ) (agent_holds ?xxx ?xxx) (and (in ?xxx ?xxx) (forall (?b - hexagonal_bin) (on ?b) ) ) ) ) )
      )
    )
    (preference preference2
      (exists (?c - block ?c - teddy_bear)
        (then
          (once (touch ?c) )
          (once (object_orientation ?c ?c) )
          (hold (in_motion rug) )
        )
      )
    )
    (forall (?i - shelf)
      (and
        (preference preference3
          (forall (?e ?t - golfball)
            (exists (?l - cube_block ?z - dodgeball)
              (then
                (once (in_motion agent side_table) )
                (once (= 1 7) )
                (once (not (and (agent_holds ?i ?i) (and (not (adjacent ?z) ) (not (rug_color_under ?i ?z) ) (on ?z) ) ) ) )
              )
            )
          )
        )
        (preference preference4
          (then
            (once (not (not (agent_holds agent) ) ) )
            (hold (agent_holds ?i top_shelf) )
            (hold (agent_holds ?i ?i) )
          )
        )
      )
    )
    (preference preference5
      (then
        (hold (in_motion ?xxx ?xxx) )
        (hold (in ?xxx ?xxx) )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (- (+ (> 3 5 )
        (count preference4:dodgeball)
      )
    )
    (count preference3:dodgeball)
  )
)
(:scoring
  4
)
)


(define (game game-id-68) (:domain medium-objects-room-v1)
(:setup
  (not
    (game-conserved
      (in_motion agent)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (in_motion ?xxx) (agent_holds ?xxx ?xxx) ) )
        (once (in_motion ?xxx) )
        (once (in_motion desk ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (> (count preference1:blue) (* (count preference1:pink) (* (count preference1:dodgeball:block) (total-score) )
      )
    )
    (or
      (not
        (or
          (not
            (or
              (or
                (or
                  (>= 6 (+ 1 (count-once-per-objects preference1:pink) )
                  )
                  (>= 7 (+ (total-score) (count-once-per-objects preference1:yellow) )
                  )
                )
                (>= (count preference1:green) (count preference1:dodgeball:purple) )
              )
              (>= 5 (count preference1:bed) )
            )
          )
          (>= (* (* (count preference1:beachball) (* (* (* (count-once-per-external-objects preference1:basketball:dodgeball) (count-once-per-objects preference1:green) )
                    (count preference1:bed)
                  )
                  5
                )
                1
                (total-time)
              )
              (count preference1:basketball)
              10
              (+ (count preference1:dodgeball) (+ 3 (* (count preference1:yellow) 2 (count preference1:golfball) )
                )
              )
            )
            5
          )
        )
      )
      (>= 3 4 )
      (>= (* (<= (count preference1:beachball) (count preference1:pink:pink) )
          300
        )
        (total-score)
      )
      (>= (total-score) (count preference1:orange:golfball) )
    )
  )
)
(:scoring
  (= (count-once-per-objects preference1:yellow_pyramid_block) 5 )
)
)


(define (game game-id-69) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (agent_holds ?xxx)
      )
    )
    (game-conserved
      (and
        (and
          (and
            (and
              (< 1 1)
            )
            (not
              (agent_holds ?xxx ?xxx)
            )
          )
          (adjacent ?xxx)
        )
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y ?b - (either cellphone cube_block) ?x - blue_pyramid_block)
        (exists (?t - pyramid_block)
          (then
            (once (not (agent_holds ?x ?x) ) )
            (hold-while (not (exists (?w - dodgeball) (not (on bed ?x) ) ) ) (and (forall (?k - flat_block ?k - dodgeball) (on ?k ?x) ) (adjacent ?x) ) )
            (once (on rug ?x) )
          )
        )
      )
    )
    (preference preference2
      (exists (?x - (either desktop dodgeball) ?r - book)
        (then
          (hold (in_motion ?r ?r ?r) )
          (hold (not (agent_holds ?r) ) )
          (once (not (not (or (in_motion ?r) (agent_holds ?r ?r) ) ) ) )
        )
      )
    )
    (preference preference3
      (exists (?o - triangular_ramp)
        (exists (?z - hexagonal_bin)
          (exists (?g - (either cellphone pen beachball dodgeball dodgeball cube_block chair))
            (then
              (once (< (distance ?o side_table agent) (distance ?o 2)) )
              (once (< 1 2) )
              (hold (exists (?v - hexagonal_bin) (not (and (not (and (and (not (not (on ?v ?v) ) ) (not (and (agent_holds ?o) (forall (?t - game_object) (not (agent_holds agent ?z) ) ) ) ) ) (adjacent_side ?o) ) ) (in ?o floor ?g) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 3 (count-unique-positions preference1:beachball) )
    (>= (count preference1:beachball) 3 )
    (> (= (* (+ 0 (count-once-per-objects preference2:green:beachball) )
          (count-once-per-objects preference1:purple)
        )
        (count-once-per-external-objects preference3:basketball:side_table)
      )
      (count preference3:golfball:pink)
    )
  )
)
(:scoring
  (count preference1:dodgeball:green)
)
)


(define (game game-id-70) (:domain few-objects-room-v1)
(:setup
  (exists (?u - curved_wooden_ramp ?b - game_object ?y - block)
    (and
      (and
        (game-conserved
          (on ?y ?y)
        )
      )
      (game-optional
        (is_setup_object bed)
      )
      (forall (?k - color)
        (exists (?b - game_object)
          (exists (?o - cylindrical_block)
            (game-optional
              (exists (?p - ball)
                (not
                  (in_motion ?k ?p)
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
    (forall (?r - cylindrical_block)
      (and
        (preference preference1
          (then
            (once (agent_holds agent) )
            (once-measure (agent_holds ?r) (distance ?r room_center ?r) )
            (hold (same_object top_shelf ?r) )
            (hold-while (agent_holds ?r) (agent_holds ?r) )
          )
        )
      )
    )
    (forall (?p - block ?u - teddy_bear)
      (and
        (preference preference2
          (then
            (hold (in ?u) )
            (hold (not (and (on yellow) (agent_holds ?u) ) ) )
          )
        )
        (preference preference3
          (then
            (once (and (and (and (not (on ?u ?u) ) (on ?u desk) ) (agent_holds ?u ?u) ) (on ?u ?u) (not (in ?u) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (= 10 (count-once-per-objects preference2:dodgeball) )
    (or
      (or
        (>= 3 (count preference3:beachball) )
        (>= (count preference2:dodgeball) 5 )
      )
      (> (count preference2:yellow) (<= (count preference3:green:basketball) 2 )
      )
    )
    (>= (* 12 4 )
      (count preference1:yellow_cube_block:yellow)
    )
  )
)
(:scoring
  (count-once-per-objects preference2:beachball)
)
)


(define (game game-id-71) (:domain few-objects-room-v1)
(:setup
  (and
    (or
      (and
        (exists (?z - dodgeball ?d - pyramid_block)
          (or
            (and
              (game-conserved
                (and
                  (< (distance ?d ?d) (distance back 9))
                  (in_motion ?d)
                )
              )
            )
            (and
              (game-conserved
                (in_motion bridge_block)
              )
              (and
                (game-optional
                  (agent_holds ?d)
                )
                (exists (?j - hexagonal_bin)
                  (and
                    (exists (?i - (either pencil golfball yellow_cube_block))
                      (forall (?k - hexagonal_bin)
                        (exists (?w - ball)
                          (game-conserved
                            (same_type rug ?i)
                          )
                        )
                      )
                    )
                    (and
                      (forall (?o - block)
                        (forall (?b - dodgeball)
                          (and
                            (game-optional
                              (agent_holds ?o bed)
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (game-optional
                  (on ?d)
                )
              )
              (game-conserved
                (adjacent ?d)
              )
            )
          )
        )
        (forall (?t - shelf ?q - chair)
          (and
            (game-conserved
              (on ?q bed)
            )
          )
        )
      )
      (and
        (game-conserved
          (in ?xxx ?xxx)
        )
      )
      (not
        (game-conserved
          (and
            (and
              (and
                (exists (?w - block)
                  (touch agent ?w)
                )
                (on ?xxx ?xxx)
              )
              (agent_holds ?xxx)
            )
            (not
              (or
                (and
                  (exists (?r - curved_wooden_ramp)
                    (not
                      (and
                        (not
                          (agent_holds ?r)
                        )
                        (not
                          (exists (?m - dodgeball ?g - cube_block ?b - dodgeball)
                            (on bed)
                          )
                        )
                        (agent_holds floor)
                      )
                    )
                  )
                  (exists (?p - hexagonal_bin)
                    (in ?p)
                  )
                  (in ?xxx ?xxx)
                  (and
                    (in ?xxx)
                    (agent_holds ?xxx ?xxx)
                  )
                )
                (in_motion rug floor)
                (in_motion ?xxx)
                (not
                  (agent_holds ?xxx west_wall floor rug)
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
    (preference preference1
      (then
        (hold (on ?xxx) )
        (once (in_motion ?xxx) )
        (once (on ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (>= 7 (count preference1:beachball:blue_pyramid_block) )
  )
)
(:scoring
  (external-forall-maximize
    (count-once-per-objects preference1:rug)
  )
)
)


(define (game game-id-72) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?i - beachball ?f - block)
      (and
        (not
          (forall (?w - hexagonal_bin ?a - cube_block)
            (exists (?e - block)
              (and
                (exists (?g - hexagonal_bin ?x - (either bridge_block wall mug alarm_clock main_light_switch dodgeball game_object) ?w - ball)
                  (game-optional
                    (agent_holds ?e ?a)
                  )
                )
              )
            )
          )
        )
      )
    )
    (game-conserved
      (adjacent ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - wall)
        (then
          (once (and (touch ?z ?z) (toggled_on ?z ?z) (= (distance 1 ?z) 1) ) )
          (hold (toggled_on ?z) )
          (once (not (agent_holds ?z ?z) ) )
        )
      )
    )
    (preference preference2
      (exists (?c - hexagonal_bin ?a - cylindrical_block)
        (exists (?h - game_object)
          (exists (?n - wall)
            (at-end
              (agent_holds ?a agent)
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
      (>= (* 18 (* (+ 2 (* (+ (count preference2:hexagonal_bin:basketball) (* (<= (count preference1:wall) (count-once-per-objects preference2:doggie_bed:pink) )
                    4
                  )
                )
                (* (count-increasing-measure preference2:hexagonal_bin) (count-once-per-objects preference1) )
              )
            )
            (count-once-per-objects preference1:hexagonal_bin)
          )
          (total-score)
        )
        (>= (* (count preference2:dodgeball) (* (* 10 (* (- 4 )
                  (count preference1:hexagonal_bin:purple)
                  (+ 7 (count-once-per-objects preference1:hexagonal_bin) )
                  (count-once preference1:pink_dodgeball)
                )
              )
              (count-once-per-external-objects preference2:dodgeball:pink)
              (count preference2:yellow)
              (count preference1:hexagonal_bin)
            )
          )
          30
        )
      )
      (>= 7 (* (* (* (count-overlapping preference2:purple) (* (total-time) 2 )
              (count-once-per-objects preference2:hexagonal_bin:green)
            )
            (count preference1:brown)
          )
          15
        )
      )
    )
  )
)
(:scoring
  30
)
)


(define (game game-id-73) (:domain few-objects-room-v1)
(:setup
  (forall (?m - (either doggie_bed wall))
    (game-conserved
      (not
        (in_motion ?m agent)
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - game_object)
      (and
        (preference preference1
          (exists (?a - dodgeball ?g - dodgeball ?y - (either dodgeball laptop))
            (exists (?c - hexagonal_bin)
              (exists (?q - bridge_block)
                (then
                  (hold-while (in_motion agent ?r) (touch ?r ?c) )
                  (hold (< (distance 0 ?q 4) 1) )
                  (hold (agent_holds ?q pink_dodgeball) )
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?h - curved_wooden_ramp)
        (exists (?v - chair)
          (then
            (hold-to-end (on ?v) )
            (hold (and (agent_holds ?v ?h) (opposite ?v ?h) ) )
            (once (in ?h ?v) )
          )
        )
      )
    )
    (preference preference3
      (exists (?n - dodgeball ?u - dodgeball ?o - hexagonal_bin)
        (exists (?q - chair ?w - (either curved_wooden_ramp golfball))
          (then
            (once (not (touch agent ?o) ) )
            (once (< 1 (distance 5 ?w)) )
            (once (not (and (in_motion ?w) (in_motion ?o ?o) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (count-once preference1:cube_block:golfball) )
)
(:scoring
  (count-shortest preference3:beachball)
)
)


(define (game game-id-74) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (< 3 (distance 0 ?xxx))
    )
    (exists (?r - hexagonal_bin ?r - hexagonal_bin)
      (exists (?e - building ?s - block ?l - hexagonal_bin ?p - hexagonal_bin ?h - hexagonal_bin)
        (game-optional
          (not
            (adjacent ?r)
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
        (preference preference1
          (then
            (hold (not (in ?c) ) )
            (once (is_setup_object rug) )
            (hold (in_motion ?c ?c) )
          )
        )
      )
    )
    (preference preference2
      (exists (?b - building)
        (exists (?m - wall ?h - block)
          (exists (?x - (either blue_cube_block pillow) ?p - dodgeball)
            (exists (?m - book ?r - color)
              (then
                (once-measure (on ?p) (distance door green_golfball) )
                (once (in_motion ?h ?p) )
                (once (in_motion ?h ?h) )
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
    (or
      (>= 40 300 )
      (>= (* (count preference2:dodgeball) 30 )
        (count preference2:top_drawer)
      )
      (not
        (<= (* (count preference1:yellow:pink_dodgeball) (/ (* 300 (count-once preference1:rug) (count-once-per-objects preference2:dodgeball:rug) )
              (* 3 3 )
            )
            10
          )
          10
        )
      )
    )
    (>= (count preference2:beachball) (count preference2:green) )
  )
)
(:scoring
  (* (external-forall-maximize (* 5 2 )
    )
    (count-once preference2:yellow)
  )
)
)


(define (game game-id-75) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (and
          (and
            (agent_holds ?xxx)
            (on ?xxx ?xxx)
          )
          (in ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - curved_wooden_ramp)
        (then
          (once (not (< (distance ?n ?n) (distance ?n ?n)) ) )
          (once (rug_color_under ?n) )
          (hold-while (agent_holds ?n agent) (and (not (not (adjacent ?n ?n) ) ) (not (and (> 2 (distance ?n back)) (object_orientation ?n ?n) ) ) ) (< 0.5 (x_position 8 8)) (agent_holds pink_dodgeball) )
        )
      )
    )
    (forall (?n ?m ?d ?z - game_object)
      (and
        (preference preference2
          (exists (?a - game_object ?q - block ?r - dodgeball ?u - wall)
            (then
              (once (agent_holds ?u) )
              (hold (on ?u) )
              (once (in_motion ?u) )
            )
          )
        )
        (preference preference3
          (then
            (hold (not (not (not (agent_holds ?m) ) ) ) )
            (once (not (exists (?a - cube_block ?w - tall_cylindrical_block) (not (in ?n ?w) ) ) ) )
            (once (on ?d) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference3:beachball) (* 1 10 )
  )
)
(:scoring
  100
)
)


(define (game game-id-76) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (or
      (in_motion ?xxx)
      (exists (?g - curved_wooden_ramp)
        (in_motion ?g)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (and (not (rug_color_under ?xxx ?xxx) ) (< 2 (distance ?xxx desk)) (and (agent_holds ?xxx ?xxx) (not (and (adjacent ?xxx) (adjacent_side ?xxx front) (exists (?s - hexagonal_bin ?r - doggie_bed ?p - drawer) (in_motion ?p left) ) (in ?xxx ?xxx) (and (agent_holds ?xxx) (touch ?xxx ?xxx) ) (agent_holds ?xxx) ) ) ) ) (forall (?t - (either cylindrical_block pyramid_block golfball) ?g - building) (not (and (and (agent_holds ?g) (and (in_motion ?g ?g) (not (and (< (x_position 3 ?g) (distance ?g)) (adjacent ?g) ) ) ) ) (and (touch floor) (in ?g ?g) ) ) ) ) ) )
        (once (exists (?a - ball) (and (in_motion desk) (< 10 1) ) ) )
        (once (on ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (= 2 (* 3 (count preference1) )
    )
    (or
      (<= (count-same-positions preference1:beachball:blue_pyramid_block:green) 15 )
      (>= (count-shortest preference1:basketball) 7 )
    )
  )
)
(:scoring
  (count-once-per-objects preference1:tan)
)
)


(define (game game-id-77) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (adjacent agent ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (and (in_motion ) (on desk ?xxx) ) (and (in_motion ball) (on ?xxx ?xxx) (adjacent agent ?xxx) (touch ?xxx south_wall) ) (not (agent_holds ?xxx) ) )
        (once (agent_holds ?xxx ?xxx) )
        (hold (not (and (and (on ?xxx ?xxx) (in_motion ?xxx ?xxx) ) ) ) )
      )
    )
    (forall (?q - cube_block)
      (and
        (preference preference2
          (then
            (once (in ?q) )
            (once (same_type ?q rug) )
            (hold (same_color desk) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (external-forall-maximize 3 ) (+ (- 6 )
        (- 2 )
      )
    )
    (>= (* (* 10 2 )
        (* 18 (* (count preference1) (* (count preference2:pink_dodgeball:beachball) (total-score) )
          )
        )
        (* 100 (count-increasing-measure preference2:dodgeball) )
        (- (and 15 ) )
      )
      (count-once-per-objects preference1:hexagonal_bin:beachball)
    )
    (>= (total-score) (external-forall-maximize 1 ) )
  )
)
(:scoring
  (+ (count preference1:tan:dodgeball) 5 )
)
)


(define (game game-id-78) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?f - block)
      (and
        (preference preference1
          (then
            (hold (and (agent_holds ?f) (on ?f) (and (on ?f desk) (agent_holds ?f ?f) ) (agent_holds pink) ) )
            (once (and (agent_holds ?f) (and (open floor) (and (touch ?f) (touch ?f ?f) ) ) ) )
            (once (not (not (in_motion ?f) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count-once preference1:blue_dodgeball) (* (or (count-once-per-external-objects preference1:beachball) (count-once-per-external-objects preference1:cube_block:yellow) ) 1 )
    )
    (or
      (= (+ (* 1 (* 1 (count-overlapping preference1:tan) )
            (count preference1:basketball)
            (+ (count-unique-positions preference1:green) (count preference1:beachball) )
            (* (count preference1:orange) (count-same-positions preference1:dodgeball:book) )
            (* (+ 1 (+ (- (* 4 (* 3 (+ (total-time) (* (not 40 ) (count-once-per-objects preference1:top_drawer:orange) )
                        )
                        (* 1 5 )
                      )
                      5
                    )
                  )
                  3
                  (total-score)
                )
              )
              (count preference1:golfball)
            )
          )
          (* (= (count preference1:pink_dodgeball:dodgeball) (count preference1:pink) )
            (count preference1:pink)
            (count-once preference1:beachball)
            10
          )
        )
        (count-once preference1:purple)
      )
      (or
        (or
          (and
            (>= (count-once preference1:pink:dodgeball:beachball) (+ (+ (+ (+ (count preference1:dodgeball:blue_pyramid_block) 5 )
                    (count-unique-positions preference1:red)
                    3
                    (count preference1:bed)
                    (external-forall-maximize
                      1
                    )
                  )
                  (* (- (+ 15 (count-shortest preference1:basketball:hexagonal_bin:yellow_cube_block) )
                    )
                    (count-once-per-objects preference1:beachball)
                  )
                )
                5
              )
            )
            (>= (count-once preference1:pink) 3 )
            (>= (count-once-per-external-objects preference1:dodgeball) (external-forall-maximize (count preference1:block) ) )
          )
          (= (count preference1) (count preference1:pink) )
          (>= (count preference1:hexagonal_bin:pyramid_block) (= 9 2 )
          )
          (< 1 (count preference1:basketball) )
        )
        (>= (* (count preference1:beachball) (* (total-score) (total-time) )
          )
          3
        )
      )
      (>= 1 (- (count preference1:golfball) )
      )
    )
  )
)
(:scoring
  (- (+ 40 (count preference1:pyramid_block) (count-once-per-objects preference1:pink) (* 3 (* (- (* (* (* (count-once preference1:basketball) 1 )
                (* 18 8 (* 3 2 )
                )
              )
              (* (count preference1:basketball) (* (- (* (count-once preference1:blue_dodgeball) )
                    (+ (+ (count preference1:beachball) 20 )
                      (count-longest preference1:pink)
                    )
                  )
                  (count preference1:golfball:orange:beachball)
                )
                (count preference1:dodgeball)
              )
            )
          )
          (count preference1:red)
        )
      )
      (or
        1
        (count-longest preference1:dodgeball:basketball)
        (+ (count-once-per-objects preference1:blue_dodgeball) (+ (* (external-forall-maximize (external-forall-maximize (>= (count preference1:purple) (* (total-score) (count preference1:white) )
                  )
                )
              )
              100
            )
            6
          )
        )
      )
      (+ (count preference1:beachball:pink) )
      (* (external-forall-maximize 5 ) )
      (count preference1:green)
      30
    )
  )
)
)


(define (game game-id-79) (:domain few-objects-room-v1)
(:setup
  (forall (?n - game_object)
    (not
      (forall (?d - dodgeball ?h - triangular_ramp ?i - hexagonal_bin)
        (game-optional
          (not
            (in_motion ?i ?n)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (not
          (in_motion floor)
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:blue_dodgeball:yellow) (count preference1:golfball) )
)
(:scoring
  (- (* (* (count preference1:blue_pyramid_block) (count preference1:alarm_clock) )
      (count-once preference1:basketball)
    )
  )
)
)


(define (game game-id-80) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - triangular_ramp ?p - pillow)
        (then
          (once (agent_holds ?p ?p) )
          (hold (agent_holds ?p ?p desk) )
        )
      )
    )
  )
)
(:terminal
  (>= (+ 40 (count preference1:basketball) (* 5 (* (- (count preference1:purple) )
          (* (external-forall-minimize 3 ) (+ 2 (* 10 (* 1 3 )
              )
            )
          )
        )
      )
      (total-time)
      (count-once-per-objects preference1:bed)
      0.5
      (total-score)
      (count-once-per-objects preference1:basketball)
      (count-same-positions preference1)
    )
    (count preference1:beachball)
  )
)
(:scoring
  (+ 2 (+ (count-measure preference1:yellow) (count-once-per-objects preference1:purple:golfball) )
  )
)
)


(define (game game-id-81) (:domain few-objects-room-v1)
(:setup
  (forall (?t - cube_block)
    (exists (?e ?z - curved_wooden_ramp)
      (exists (?v - teddy_bear)
        (and
          (exists (?i - chair)
            (game-optional
              (toggled_on ?i)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (and (not (agent_holds ?xxx) ) (not (agent_holds ?xxx) ) ) ) )
        (hold-while (adjacent bed) (opposite ?xxx) (agent_holds ?xxx) (on ?xxx) )
        (once (not (and (agent_holds pink_dodgeball) (< 1 (distance agent ?xxx)) (in ?xxx) ) ) )
        (once (in_motion ?xxx) )
        (once (and (< 1 (distance 6 4)) (and (agent_holds ?xxx ?xxx) (< (distance ?xxx ?xxx) (distance 3 ?xxx)) ) (on ?xxx ?xxx) ) )
        (once (not (not (< (distance room_center ?xxx) (distance ?xxx)) ) ) )
      )
    )
  )
)
(:terminal
  (>= (* (- (+ 6 20 )
      )
      (count preference1:hexagonal_bin)
    )
    (count preference1:yellow)
  )
)
(:scoring
  (* (count-once-per-objects preference1:book:yellow) (* 3 (external-forall-maximize (count-once-per-external-objects preference1:pink) ) )
  )
)
)


(define (game game-id-82) (:domain few-objects-room-v1)
(:setup
  (and
    (or
      (and
        (and
          (game-optional
            (< (distance ?xxx ?xxx) 2)
          )
        )
        (and
          (and
            (game-optional
              (exists (?j - building ?j - desk_shelf ?e - wall ?s - chair)
                (not
                  (on agent)
                )
              )
            )
            (game-conserved
              (in_motion ?xxx)
            )
            (and
              (and
                (and
                  (game-conserved
                    (and
                      (in_motion ?xxx)
                      (and
                        (not
                          (< (distance 7 ?xxx) (distance agent ?xxx))
                        )
                        (or
                          (and
                            (and
                              (touch ?xxx ?xxx)
                              (and
                                (adjacent_side ?xxx ?xxx)
                                (not
                                  (not
                                    (and
                                      (and
                                        (not
                                          (not
                                            (and
                                              (adjacent_side ?xxx ?xxx)
                                              (not
                                                (not
                                                  (not
                                                    (in pink ?xxx)
                                                  )
                                                )
                                              )
                                            )
                                          )
                                        )
                                        (in ?xxx)
                                        (and
                                          (in agent)
                                          (on ?xxx)
                                        )
                                        (in_motion bed)
                                        (on ?xxx)
                                        (not
                                          (not
                                            (and
                                              (agent_holds bed)
                                              (and
                                                (agent_holds ?xxx ?xxx)
                                                (adjacent ?xxx)
                                              )
                                            )
                                          )
                                        )
                                      )
                                      (not
                                        (on ?xxx front)
                                      )
                                    )
                                  )
                                )
                              )
                            )
                            (and
                              (agent_holds ?xxx upright)
                              (exists (?t - (either golfball lamp))
                                (adjacent ?t)
                              )
                              (and
                                (>= (distance_side room_center ?xxx) (distance ?xxx ?xxx))
                                (exists (?j - dodgeball)
                                  (and
                                    (in ?j ?j)
                                    (in_motion ?j pink_dodgeball)
                                    (and
                                      (on ?j)
                                      (and
                                        (or
                                          (and
                                            (not
                                              (in_motion ?j ?j)
                                            )
                                            (in_motion ?j)
                                          )
                                          (not
                                            (agent_holds ?j)
                                          )
                                        )
                                        (in_motion front)
                                      )
                                    )
                                  )
                                )
                                (not
                                  (agent_holds ?xxx)
                                )
                                (and
                                  (not
                                    (adjacent pink_dodgeball)
                                  )
                                  (and
                                    (not
                                      (object_orientation agent ?xxx)
                                    )
                                    (in ?xxx ?xxx)
                                  )
                                  (and
                                    (in_motion ?xxx)
                                    (< 7 1)
                                  )
                                )
                              )
                            )
                          )
                          (not
                            (and
                              (not
                                (and
                                  (not
                                    (in_motion ?xxx floor)
                                  )
                                  (not
                                    (in_motion ?xxx ?xxx)
                                  )
                                )
                              )
                              (and
                                (on ?xxx)
                                (and
                                  (on ?xxx)
                                  (or
                                    (not
                                      (not
                                        (and
                                          (not
                                            (and
                                              (same_type ?xxx)
                                              (> 1 3)
                                            )
                                          )
                                          (agent_holds ?xxx ?xxx)
                                          (and
                                            (adjacent ?xxx)
                                            (not
                                              (on top_drawer ?xxx)
                                            )
                                          )
                                        )
                                      )
                                    )
                                    (in_motion ?xxx ?xxx)
                                  )
                                  (touch ?xxx ?xxx)
                                )
                              )
                              (agent_holds ?xxx ?xxx)
                            )
                          )
                        )
                      )
                    )
                  )
                  (and
                    (game-conserved
                      (or
                        (same_color ?xxx)
                        (and
                          (and
                            (is_setup_object ?xxx ?xxx)
                            (< (distance room_center ?xxx) (distance ?xxx ?xxx))
                          )
                          (on ?xxx)
                          (exists (?d - hexagonal_bin ?c - cylindrical_block ?b - cube_block)
                            (touch ?b)
                          )
                        )
                      )
                    )
                  )
                  (game-optional
                    (and
                      (and
                        (not
                          (adjacent ?xxx ?xxx)
                        )
                        (in_motion rug ?xxx)
                        (in_motion ?xxx ?xxx)
                      )
                      (not
                        (not
                          (not
                            (not
                              (in_motion ?xxx rug)
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
            (not
              (and
                (agent_holds ?xxx)
                (in_motion ?xxx ?xxx)
                (and
                  (in_motion rug)
                  (agent_holds ?xxx ?xxx)
                )
                (and
                  (agent_holds rug ?xxx)
                  (opposite ?xxx)
                  (game_over ?xxx)
                )
              )
            )
          )
          (exists (?d - (either dodgeball pyramid_block))
            (game-conserved
              (in_motion ?d)
            )
          )
        )
      )
    )
    (not
      (exists (?c - ball)
        (game-conserved
          (forall (?h - drawer ?e - doggie_bed)
            (not
              (agent_holds ?c ?c)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (and
          (touch ?xxx)
          (agent_holds ?xxx)
        )
      )
    )
    (preference preference2
      (exists (?m - hexagonal_bin)
        (exists (?r - hexagonal_bin ?j - curved_wooden_ramp)
          (then
            (once (agent_holds ?m) )
            (hold (agent_holds desk) )
            (once (and (and (not (and (not (agent_holds ?j) ) (in_motion ?m) ) ) ) ) )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (and
          (same_object pink_dodgeball)
          (not
            (and
              (and
                (on ?xxx ?xxx ?xxx)
                (exists (?s - shelf)
                  (not
                    (in_motion ?s)
                  )
                )
              )
              (between ?xxx)
            )
          )
        )
      )
    )
    (forall (?l - (either cylindrical_block rug) ?z ?m - ball)
      (and
        (preference preference4
          (then
            (hold (in_motion ?m ?z) )
            (any)
            (once (in_motion ?m ?m) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference3:basketball) (and (* (count preference4:book) 4 (count preference3:red) (+ 30 (+ (* (* 6 (count-once preference4:pink) )
            )
            4
          )
        )
      )
      (= (* 3 4 )
        6
      )
    )
  )
)
(:scoring
  4
)
)


(define (game game-id-83) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (= 5 (distance ?xxx agent))
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - shelf ?e - hexagonal_bin)
        (exists (?l - hexagonal_bin)
          (at-end
            (touch ?l)
          )
        )
      )
    )
    (preference preference2
      (exists (?m - yellow_pyramid_block ?w - (either book yellow_cube_block))
        (exists (?l ?i - dodgeball)
          (then
            (once (< 9 (distance door bed)) )
            (once (= (distance ?l ?l) (distance ?i ?w)) )
            (once (adjacent ?w) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (count-once preference1:book) )
)
(:scoring
  2
)
)


(define (game game-id-84) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (and
      (and
        (or
          (and
            (in ?xxx)
            (in ?xxx)
          )
          (not
            (and
              (and
                (on ?xxx)
                (and
                  (agent_holds ?xxx ?xxx)
                  (in_motion ?xxx)
                )
                (not
                  (agent_holds ?xxx)
                )
              )
            )
          )
        )
        (or
          (on block ?xxx)
          (in_motion ?xxx)
        )
      )
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (adjacent upright) )
        (hold (not (not (not (in_motion ?xxx) ) ) ) )
        (hold (agent_holds desk ?xxx) )
      )
    )
    (preference preference2
      (exists (?z - doggie_bed)
        (exists (?d - (either key_chain desktop key_chain yellow_cube_block cd yellow doggie_bed))
          (exists (?x - (either golfball tall_cylindrical_block) ?a - teddy_bear)
            (then
              (once (not (not (agent_holds ?z) ) ) )
              (once (not (not (and (on ?d ?z) (adjacent ?z ?d) ) ) ) )
              (hold (agent_holds ?z ?d) )
              (hold (agent_crouches side_table) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:dodgeball) (* (count-shortest preference2:alarm_clock) (* 3 (count-once-per-objects preference1) (count-once-per-objects preference1:yellow) )
      )
    )
    (or
      (>= 3 (count-once-per-objects preference2:dodgeball) )
      (<= (* (total-score) (count-overlapping preference1:dodgeball:golfball) )
        15
      )
    )
  )
)
(:scoring
  (count preference1:red)
)
)


(define (game game-id-85) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (and
          (not
            (object_orientation floor)
          )
        )
      )
      (game-conserved
        (or
          (and
            (agent_holds ?xxx)
            (and
              (not
                (in_motion ?xxx ?xxx)
              )
              (and
                (in ?xxx)
                (not
                  (in_motion ?xxx)
                )
                (not
                  (or
                    (adjacent ?xxx)
                    (on ?xxx pink)
                  )
                )
                (adjacent ?xxx ?xxx)
                (in_motion ?xxx)
              )
              (or
                (not
                  (not
                    (< (distance ?xxx ?xxx) (distance green_golfball))
                  )
                )
                (agent_holds ?xxx)
                (agent_holds ?xxx ?xxx)
              )
            )
          )
          (and
            (in ?xxx)
            (is_setup_object ?xxx)
          )
        )
      )
      (exists (?d - (either tall_cylindrical_block golfball) ?g - tall_cylindrical_block)
        (and
          (and
            (game-optional
              (not
                (agent_holds ?g)
              )
            )
          )
          (forall (?e - hexagonal_bin)
            (game-conserved
              (not
                (not
                  (in_motion ?e agent)
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
    (preference preference1
      (then
        (hold (and (and (and (in_motion ?xxx) (on ?xxx) ) (on agent ?xxx) (and (or (on ?xxx ?xxx) (touch ?xxx) ) (> 2 1) ) (not (in_motion ?xxx ?xxx) ) ) (agent_holds ?xxx) (in_motion ?xxx) ) )
        (once (agent_holds ?xxx ?xxx) )
        (once (adjacent_side ?xxx) )
      )
    )
    (preference preference2
      (exists (?a ?y - block ?p ?o - drawer)
        (exists (?k - game_object)
          (then
            (hold (agent_holds agent) )
            (hold (not (not (exists (?t - hexagonal_bin) (in ?o) ) ) ) )
            (once (adjacent ?k) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (external-forall-maximize (* (* 3 (= (count-once preference1:yellow) )
            0.5
          )
          3
        )
      )
      (count preference2:yellow)
    )
    (or
      (or
        (> (/ 6 (+ 5 (+ 6 (total-score) (count-once-per-objects preference2:beachball) )
            )
          )
          (external-forall-maximize
            5
          )
        )
        (or
          (>= (* 2 (external-forall-maximize (* (- 6 )
                  1
                )
              )
              (>= (* 100 (- (+ (+ (* 1 (+ (count-once-per-objects preference1:purple:dodgeball) (count-once-per-objects preference1:pink:blue_pyramid_block) )
                        )
                        2
                      )
                      (* (total-time) (count preference1:green) (+ (* 10 2 )
                          (* (* 2 (count-overlapping preference1:blue_dodgeball) )
                            (>= 3 (count preference1:green:yellow) )
                          )
                        )
                      )
                    )
                  )
                )
                (* 25 4 )
              )
              5
              3
              (* (count-shortest preference1:yellow_cube_block:yellow) (count preference1:pink) )
              6
              (count preference2:beachball)
              5
            )
            (total-score)
          )
          (> 0 3 )
        )
      )
    )
    (>= (count preference1:green) (count-unique-positions preference2:basketball:beachball) )
  )
)
(:scoring
  (count preference2:green)
)
)


(define (game game-id-86) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (in_motion desk) )
        (hold (in_motion ?xxx) )
        (once-measure (agent_holds ?xxx) (distance 3 ?xxx) )
      )
    )
    (forall (?l - shelf)
      (and
        (preference preference2
          (then
            (hold-while (not (adjacent ?l) ) (in_motion ?l ?l) )
            (once (adjacent ?l ?l) )
            (once (agent_holds ?l ?l) )
          )
        )
        (preference preference3
          (then
            (once (agent_holds agent) )
            (hold-while (and (not (and (not (and (in_motion ?l) (not (and (in_motion ?l ?l) (in agent) ) ) ) ) (in ?l) ) ) (not (agent_holds ?l green) ) ) (agent_holds ?l ?l) )
            (once (not (between ?l ?l) ) )
          )
        )
      )
    )
    (forall (?f - hexagonal_bin)
      (and
        (preference preference4
          (exists (?w - wall)
            (exists (?o - (either doggie_bed curved_wooden_ramp) ?l - dodgeball ?i - pillow)
              (then
                (once (< (distance ?w ?f) 0) )
                (hold (not (and (or (agent_holds ?w ?w) (not (in_motion ?f ?i) ) ) (same_type desk ?w) (exists (?g - hexagonal_bin) (and (adjacent ?f) (and (not (forall (?e ?x - hexagonal_bin) (agent_holds ?w) ) ) (not (agent_holds ?f ?w) ) ) (not (in_motion agent) ) ) ) ) ) )
                (hold (rug_color_under ?i) )
              )
            )
          )
        )
        (preference preference5
          (exists (?p - doggie_bed ?j - doggie_bed)
            (exists (?y - block ?u - block)
              (then
                (any)
                (hold-while (and (in_motion ?j ?u) (< 1 (distance 6 ?u)) ) (and (agent_holds ?j) (in ?u ?f) ) )
                (once (above ?f) )
                (once (in_motion ?u) )
                (once (adjacent ?f south_wall) )
                (once (not (exists (?i - dodgeball ?x - building) (agent_holds ?j) ) ) )
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
    (> (count-once-per-external-objects preference3:beachball:dodgeball) (count-increasing-measure preference4:dodgeball) )
    (>= (count preference5:yellow_cube_block) 4 )
  )
)
(:scoring
  (/
    (count-once-per-objects preference1:alarm_clock:dodgeball)
    (* 5 (count-once-per-objects preference4:blue_dodgeball) (total-time) 6 2 30 )
  )
)
)


(define (game game-id-87) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (touch ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - hexagonal_bin)
        (exists (?j - ball)
          (exists (?a - dodgeball)
            (exists (?v - (either cd cellphone) ?k - dodgeball ?i - building)
              (exists (?o ?y - teddy_bear)
                (exists (?s - hexagonal_bin ?g - hexagonal_bin ?p ?r ?x - dodgeball)
                  (exists (?g - dodgeball ?l - hexagonal_bin ?h - hexagonal_bin)
                    (then
                      (once (on ?x) )
                      (once (exists (?e - shelf) (< 1 (distance ?a)) ) )
                      (once (agent_holds ?z ?i) )
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
  (>= (- 5 )
    (count-once preference1:purple)
  )
)
(:scoring
  (count-once-per-objects preference1:basketball:blue_cube_block)
)
)


(define (game game-id-88) (:domain few-objects-room-v1)
(:setup
  (forall (?k - hexagonal_bin)
    (exists (?u - (either cube_block pyramid_block) ?s - red_pyramid_block)
      (and
        (and
          (and
            (game-conserved
              (in ?s ?k ?s)
            )
          )
          (game-conserved
            (in_motion ?k bed)
          )
          (not
            (game-conserved
              (adjacent ?s)
            )
          )
        )
        (game-optional
          (agent_holds floor ?k)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - hexagonal_bin)
      (and
        (preference preference1
          (exists (?g - game_object)
            (exists (?v - shelf ?c - hexagonal_bin)
              (at-end
                (and
                  (agent_holds ?q)
                  (or
                    (not
                      (in ?q)
                    )
                    (between ?c ?g)
                  )
                )
              )
            )
          )
        )
        (preference preference2
          (then
            (once (in ?q) )
            (once (agent_holds ?q ?q) )
            (hold (and (on ?q tan) (and (and (agent_holds ?q) (in_motion ?q ?q) ) (and (not (agent_holds ?q) ) (in_motion ?q ?q) ) ) (in ?q agent) (and (< (distance 6 9) (distance ?q 10)) (not (in_motion ?q ?q) ) ) ) )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (agent_holds ?xxx ?xxx)
      )
    )
    (forall (?p - dodgeball ?r - (either golfball yellow book))
      (and
        (preference preference4
          (then
            (once (in ?r ?r) )
            (once (not (or (and (and (opposite floor) (in_motion ?r) (not (not (and (adjacent_side ?r ?r) (not (touch ?r) ) ) ) ) ) (and (and (agent_holds ?r ?r) (not (in ?r ?r ?r) ) ) (agent_holds ?r) (in ?r ?r) ) ) (in_motion ?r) ) ) )
            (once (object_orientation ?r ?r) )
          )
        )
        (preference preference5
          (at-end
            (agent_holds ?r)
          )
        )
        (preference preference6
          (exists (?d - pillow)
            (then
              (once (and (not (and (not (< (building_size 8 ?r) 1) ) (and (exists (?o - (either basketball dodgeball dodgeball)) (agent_holds ?r) ) (not (or (and (not (not (in_motion bed) ) ) (in floor) ) (on ?d ?r) (in_motion rug) ) ) (not (not (in_motion ?d ?d) ) ) ) ) ) (in ?d ?d) ) )
              (once (not (and (in_motion ?r upright) (not (and (agent_holds ?d) (on pink_dodgeball) (not (exists (?p - (either dodgeball alarm_clock) ?p - color) (in_motion ?p) ) ) (and (not (= (distance agent ?d) 4) ) (and (agent_holds ?r ?r) (on ?r) (agent_holds ?d door) (adjacent ?d) (not (in ) ) (exists (?x - cube_block) (not (in_motion agent ?x) ) ) ) ) ) ) ) ) )
              (once (in_motion ?r ?d) )
            )
          )
        )
      )
    )
    (preference preference7
      (then
        (once (agent_holds ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
        (once (and (in_motion ?xxx) (agent_holds ?xxx rug) ) )
      )
    )
    (preference preference8
      (then
        (once (touch ?xxx ?xxx) )
        (hold-while (forall (?r - chair) (agent_holds ?r floor) ) (in ?xxx ?xxx) )
        (hold (not (adjacent ?xxx) ) )
      )
    )
  )
)
(:terminal
  (or
    (or
      (>= (count-same-positions preference2:golfball) 12 )
      (> (count-once preference4:beachball) 5 )
    )
    (or
      (>= 7 2 )
      (not
        (= (* (count-once-per-objects preference4:alarm_clock) (+ (count preference2:doggie_bed) (* 5 0 (count preference6:beachball:tall_cylindrical_block) (* (= (+ (count preference3:beachball) (* 3 (count preference7:golfball) )
                    )
                    (count preference1:yellow)
                  )
                  10
                )
                (+ (count preference6:golfball) (count preference8:basketball:blue_pyramid_block) )
                (* (* (count preference2:dodgeball) (external-forall-maximize (count-once-per-objects preference6:block) ) )
                  (count preference8:blue_dodgeball:basketball)
                )
              )
            )
          )
          (/
            (count-once-per-objects preference4:pink)
            (* (* (+ (+ 5 10 )
                  (external-forall-maximize
                    (count preference7:beachball)
                  )
                )
                (count-once-per-objects preference2:green)
              )
              (and
                (* 2 (+ (count-once-per-objects preference4:basketball:alarm_clock) (- (count preference7:blue_cube_block) )
                  )
                  (count preference1:red)
                  (count preference6:triangle_block)
                  (count preference6:yellow_cube_block)
                  2
                )
                (count preference5:dodgeball)
                (count preference6:dodgeball)
              )
            )
          )
        )
      )
    )
  )
)
(:scoring
  (count-once-per-objects preference2:beachball)
)
)


(define (game game-id-89) (:domain medium-objects-room-v1)
(:setup
  (exists (?k - (either yellow_cube_block cube_block cube_block) ?q - block)
    (and
      (forall (?f - chair)
        (exists (?c - triangular_ramp)
          (game-conserved
            (on ?f ?c)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - ball)
        (exists (?m - hexagonal_bin)
          (exists (?y - block)
            (then
              (hold (not (<= 1 (distance room_center ?m)) ) )
              (once (and (agent_holds ?m ?y) (not (agent_holds ?e) ) ) )
              (hold (adjacent ?e ?e) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:basketball) (+ 10 (count-once-per-objects preference1:book) 3 10 )
  )
)
(:scoring
  5
)
)


(define (game game-id-90) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (agent_holds ?xxx)
      )
    )
    (preference preference2
      (then
        (once (exists (?c ?m - game_object) (in ?m ?c) ) )
        (once (agent_holds ?xxx) )
        (hold (and (or (on ?xxx) (not (touch ?xxx) ) ) (<= 0.5 (distance ?xxx side_table)) ) )
      )
    )
  )
)
(:terminal
  (or
    (> (not (* (* 2 (count preference2:cylindrical_block) )
          (* (count preference2:red) (- (+ (count preference2:yellow) (* 0 (count preference2:basketball:beachball:doggie_bed) 0 )
              )
            )
            30
            (count-once-per-objects preference1:basketball:doggie_bed)
            4
            15
          )
          3
        )
      )
      (* 5 )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-91) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion ?xxx bed)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (and (and (in_motion ?xxx desk) (agent_holds ?xxx) ) (and (not (in_motion ?xxx ?xxx) ) (and (touch ?xxx) (forall (?k - game_object) (agent_holds ?k) ) ) ) ) (and (in_motion ?xxx ball) (and (and (and (agent_holds ?xxx ?xxx) (not (not (or (< 1 (distance room_center 1)) (agent_holds ?xxx ?xxx) ) ) ) ) (agent_holds ?xxx ?xxx) (and (on ?xxx) (touch ?xxx floor) ) (on agent) ) (adjacent bed ?xxx) ) ) (exists (?r - doggie_bed) (on ?r) ) )
        (hold (not (in_motion ?xxx) ) )
        (hold (not (agent_holds front) ) )
      )
    )
    (preference preference2
      (then
        (hold (= 1 (distance ?xxx ?xxx)) )
        (hold (not (not (on agent ?xxx) ) ) )
        (once (< 1 1) )
      )
    )
    (preference preference3
      (exists (?h - (either key_chain laptop) ?x - cube_block ?j - blinds ?s - dodgeball ?n - hexagonal_bin)
        (exists (?e - cube_block)
          (exists (?q - hexagonal_bin)
            (exists (?u - doggie_bed)
              (exists (?c - hexagonal_bin)
                (forall (?t - hexagonal_bin)
                  (exists (?v - hexagonal_bin)
                    (at-end
                      (not
                        (< 6 1)
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
    (preference preference4
      (exists (?j - dodgeball ?g - game_object)
        (at-end
          (agent_holds ?g)
        )
      )
    )
  )
)
(:terminal
  (>= (- (>= 10 (count-once-per-objects preference4:red) )
    )
    (or
      (* (count preference3:beachball) (count preference3:basketball) (count preference1:doggie_bed:red) (external-forall-maximize (* (+ (* (count-once-per-objects preference3:golfball) 9 )
              (external-forall-maximize
                (> (count-once-per-external-objects preference4:dodgeball) 2 )
              )
            )
            (count-once-per-objects preference1:hexagonal_bin)
          )
        )
      )
    )
  )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-92) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n ?a ?p - wall)
        (at-end
          (not
            (not
              (in ?p ?p)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* (+ (count preference1:golfball:dodgeball) (count-increasing-measure preference1:basketball:green) (* (* (* 3 )
              (* (* 4 1 )
                (count-once-per-objects preference1:dodgeball)
              )
            )
            (+ (count preference1:dodgeball) (* 3 10 )
            )
          )
          (count-once preference1:basketball)
          (count-once-per-objects preference1:cube_block)
        )
        (count preference1:green)
      )
      (* (count preference1:pink:basketball) (count-once-per-objects preference1:dodgeball:hexagonal_bin) )
    )
    (count preference1:pink)
  )
)
(:scoring
  5
)
)


(define (game game-id-93) (:domain few-objects-room-v1)
(:setup
  (forall (?e - (either cd) ?b - book ?v - wall ?g - dodgeball)
    (and
      (and
        (and
          (and
            (exists (?w ?t ?o - cube_block ?d - hexagonal_bin)
              (forall (?r ?h - ball)
                (game-conserved
                  (and
                    (open ?d ?r)
                    (equal_z_position floor)
                  )
                )
              )
            )
          )
        )
        (game-conserved
          (in_motion ?g ?g)
        )
      )
      (or
        (exists (?n - block ?b - building ?b - hexagonal_bin ?h - teddy_bear)
          (forall (?s - doggie_bed ?s - dodgeball)
            (and
              (exists (?a - building)
                (forall (?n - hexagonal_bin ?f ?j - triangular_ramp)
                  (and
                    (game-conserved
                      (and
                        (not
                          (in_motion ?j)
                        )
                        (on ?a)
                      )
                    )
                    (forall (?l - doggie_bed)
                      (game-conserved
                        (and
                          (agent_holds bed)
                          (faces ?g green_golfball)
                          (not
                            (in_motion ?f)
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
          (in_motion ?g ?g)
        )
        (game-conserved
          (object_orientation ?g)
        )
      )
      (and
        (and
          (forall (?o - building)
            (exists (?w - cube_block ?i - hexagonal_bin ?p - (either teddy_bear mug))
              (game-optional
                (and
                  (not
                    (on ?o)
                  )
                  (and
                    (on ?o ?g)
                    (agent_holds ?g)
                    (not
                      (and
                        (agent_holds ?p)
                        (and
                          (and
                            (< (distance ?o 0) 1)
                            (in_motion rug ?o)
                            (and
                              (in ?o)
                              (<= 1 0.5)
                              (on ?o)
                            )
                          )
                          (adjacent ?g)
                        )
                      )
                    )
                    (forall (?x - hexagonal_bin)
                      (agent_holds agent ?g)
                    )
                    (not
                      (and
                        (in_motion ?p ?g)
                        (agent_holds desk)
                        (in_motion ?p ?p ?o)
                        (not
                          (in_motion ?o)
                        )
                        (agent_holds ?p)
                        (and
                          (not
                            (above agent ?g)
                          )
                          (in_motion upright)
                        )
                      )
                    )
                    (in_motion ?o ?p)
                    (not
                      (on ?o)
                    )
                    (adjacent ?g ?g)
                    (on agent)
                    (agent_holds ?o ?p)
                  )
                )
              )
            )
          )
          (exists (?z - teddy_bear)
            (not
              (exists (?i - dodgeball)
                (game-optional
                  (not
                    (and
                      (on ?i)
                      (forall (?m - hexagonal_bin)
                        (exists (?w - doggie_bed ?n - flat_block ?k - ball)
                          (object_orientation ?m)
                        )
                      )
                      (and
                        (forall (?u ?h - curved_wooden_ramp)
                          (< 2 3)
                        )
                        (not
                          (and
                            (or
                              (agent_holds ?g ?g)
                              (agent_holds ?z ?g)
                            )
                            (not
                              (agent_holds ?g)
                            )
                            (exists (?p - teddy_bear)
                              (not
                                (in ?z ?g)
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
          (and
            (game-conserved
              (opposite ?g ?g)
            )
            (game-optional
              (agent_holds agent)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - block ?x - block ?i - ball)
        (then
          (hold (or (agent_holds rug ?i) ) )
          (once (not (agent_holds floor ?i ?i) ) )
        )
      )
    )
    (preference preference2
      (then
        (hold (and (and (< 2 (distance room_center agent)) (not (not (in_motion ?xxx ?xxx) ) ) ) (in_motion ?xxx) (in ?xxx ?xxx) ) )
        (once (agent_holds ?xxx ?xxx) )
        (once-measure (in ?xxx ?xxx) (building_size room_center ?xxx) )
        (once (and (on ?xxx) (and (agent_holds ?xxx ?xxx) (exists (?j - curved_wooden_ramp) (and (and (and (and (and (in ?j ?j) (< 0.5 (distance ?j ?j)) ) (between ?j ?j) ) (between ?j ?j) (adjacent ?j ?j) (in_motion rug) ) (not (< 8 (distance 6 ?j)) ) (and (and (agent_holds bed) (or (adjacent_side ?j ?j) (forall (?z - (either bridge_block key_chain) ?t - hexagonal_bin) (adjacent ?t ?j) ) ) ) (and (object_orientation ?j) (not (between ?j) ) ) ) ) (and (agent_holds desk ?j) (not (and (and (or (agent_holds ?j ?j) ) (not (not (in_motion ?j desk) ) ) ) (not (agent_holds bed) ) ) ) ) ) ) ) ) )
      )
    )
  )
)
(:terminal
  (>= 1 (count preference1:yellow) )
)
(:scoring
  3
)
)


(define (game game-id-94) (:domain many-objects-room-v1)
(:setup
  (exists (?b - hexagonal_bin)
    (exists (?a - hexagonal_bin ?v - (either cube_block top_drawer) ?k - game_object)
      (exists (?f - (either pyramid_block laptop golfball))
        (and
          (game-conserved
            (and
              (on ?k ?f)
              (agent_holds ?b bed)
            )
          )
          (game-conserved
            (and
              (not
                (in_motion agent)
              )
              (not
                (same_type ?f ?f)
              )
              (not
                (agent_holds ?f rug)
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
    (forall (?g - hexagonal_bin)
      (and
        (preference preference1
          (exists (?f - ball ?l - game_object)
            (then
              (once (agent_holds ?l agent) )
              (once (and (agent_holds ?l ?l) (in ?g) ) )
              (hold (agent_holds ?l) )
            )
          )
        )
        (preference preference2
          (then
            (hold (agent_holds ?g) )
            (once (adjacent upright ?g) )
            (hold-while (and (agent_holds agent) (adjacent ?g) (agent_holds ?g) ) (on ?g) )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (and (and (< 4 1) (adjacent ?xxx) ) (not (agent_holds ?xxx ?xxx) ) ) )
        (hold (same_object ?xxx ?xxx) )
        (hold-while (adjacent rug) (and (and (not (agent_holds ?xxx ?xxx) ) (and (and (agent_holds ?xxx ?xxx) (same_color ?xxx bed) ) (agent_holds ?xxx agent) ) ) (on ?xxx ?xxx) ) (and (and (in_motion ?xxx) (touch bed ?xxx) ) ) )
      )
    )
    (preference preference4
      (at-end
        (touch ?xxx)
      )
    )
    (preference preference5
      (exists (?g - game_object)
        (then
          (once (in_motion ?g ?g) )
          (hold-while (not (and (is_setup_object ?g ?g) (on ?g) ) ) (agent_holds ?g) )
          (hold (in_motion upright ?g) )
        )
      )
    )
    (preference preference6
      (then
        (once (in_motion ?xxx ?xxx) )
        (once (and (agent_holds agent south_west_corner) (not (< 6 (distance ?xxx room_center)) ) (in_motion ?xxx) ) )
        (once (and (and (agent_holds ?xxx ?xxx) (not (not (not (not (not (forall (?x - hexagonal_bin) (not (not (on ?x) ) ) ) ) ) ) ) ) ) (agent_holds ?xxx) ) )
      )
    )
    (preference preference7
      (then
        (hold (not (agent_holds ?xxx) ) )
        (once (above ?xxx) )
        (once (and (not (and (and (object_orientation ?xxx ?xxx) (on ?xxx) ) (and (agent_holds ?xxx ?xxx) (adjacent ?xxx) ) ) ) (agent_holds ?xxx floor) ) )
      )
    )
  )
)
(:terminal
  (>= (+ (not 8 ) (* (* (count preference1:hexagonal_bin) (count-shortest preference1:dodgeball) )
        (count preference4:hexagonal_bin)
        (+ 5 )
        1
        12
        (count preference6:yellow)
      )
    )
    (+ 100 (- (count-once-per-external-objects preference5:dodgeball) )
    )
  )
)
(:scoring
  (* 5 9 )
)
)


(define (game game-id-95) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (agent_holds ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?j - hexagonal_bin ?k - curved_wooden_ramp ?l - red_dodgeball)
      (and
        (preference preference1
          (exists (?r - cube_block ?k - chair)
            (exists (?u - block)
              (exists (?t - dodgeball ?f - hexagonal_bin ?q - hexagonal_bin)
                (exists (?n - game_object ?m - hexagonal_bin ?d - hexagonal_bin ?d ?e ?z - cube_block)
                  (then
                    (once (or (on ?e rug) (and (not (in_motion ?z) ) (not (in_motion ?u) ) ) ) )
                    (once (< (distance ?k ?q) 1) )
                    (once (agent_holds ?l) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (and (and (= (distance ?xxx 8) 1) (adjacent desktop) ) (agent_holds ?xxx) ) )
        (once (< 1 1) )
        (once (and (on desk) (not (not (not (not (and (agent_holds ?xxx) (not (< 1 (distance 5 ?xxx)) ) ) ) ) ) ) ) )
      )
    )
  )
)
(:terminal
  (or
    (>= 0 (count-once-per-objects preference2:yellow_cube_block) )
  )
)
(:scoring
  (+ (* 30 (* (count-unique-positions preference1:yellow_cube_block) (external-forall-minimize 6 ) )
    )
    (+ (* (count preference1:red:dodgeball) (count preference2:purple:dodgeball) )
      (count preference2:basketball:dodgeball)
    )
  )
)
)


(define (game game-id-96) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (> 2 (distance 2 4))
      (agent_holds ?xxx ?xxx)
      (not
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - chair)
      (and
        (preference preference1
          (then
            (once (in_motion ?a) )
            (hold (exists (?x - cube_block ?y - pyramid_block) (agent_holds ?a) ) )
            (once (not (not (agent_holds ?a) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (or
      (not
        (>= (> (and (< 2 (* (total-score) 5 )
              )
              (count preference1:yellow)
              (total-score)
            )
            (* (* 0 (* (* 15 (* (* 0 (count preference1:beachball:orange) )
                      (* (* 2 (- 4 )
                        )
                        (+ (* (count preference1:dodgeball:beachball:beachball) 10 )
                          20
                          (count preference1:triangle_block:beachball)
                        )
                      )
                      5
                      (count preference1:beachball)
                      (count preference1:doggie_bed:golfball:blue_dodgeball)
                    )
                  )
                  (count preference1:blue_dodgeball:dodgeball:pink_dodgeball)
                )
              )
              5
            )
          )
          3
        )
      )
    )
    (not
      (= (count-measure preference1:dodgeball) (count preference1) )
    )
    (or
      (>= (+ 1 (+ (count-once-per-objects preference1:pink_dodgeball) (* (count preference1:beachball) (count preference1:yellow:dodgeball) )
          )
        )
        4
      )
      (>= 2 (count preference1:basketball:pink) )
      (>= 10 5 )
    )
  )
)
(:scoring
  (+ (count-once-per-objects preference1:dodgeball) 30 (- (count-total preference1:pink) )
  )
)
)


(define (game game-id-97) (:domain few-objects-room-v1)
(:setup
  (exists (?a - cube_block ?a ?f ?p - dodgeball)
    (and
      (and
        (game-conserved
          (in_motion ?p)
        )
        (forall (?h - drawer)
          (and
            (game-conserved
              (in ?p)
            )
            (game-conserved
              (in_motion agent ?h)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - hexagonal_bin)
        (exists (?d - pillow)
          (then
            (hold-while (agent_holds ?b) (and (in ?b) ) )
            (once (and (< (distance ?d 10) 0) ) )
            (once (not (agent_holds ?b) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (= (* 5 (count-shortest preference1:dodgeball) )
    (count preference1:red)
  )
)
(:scoring
  (* (and (count-once-per-objects preference1:golfball:dodgeball:blue_dodgeball) (- 5 )
    )
    (count-once-per-external-objects preference1:yellow)
    (count preference1:beachball)
  )
)
)


(define (game game-id-98) (:domain few-objects-room-v1)
(:setup
  (exists (?y - hexagonal_bin)
    (exists (?w - beachball ?j - block)
      (and
        (forall (?f - hexagonal_bin)
          (or
            (not
              (forall (?r - bridge_block)
                (exists (?z - dodgeball)
                  (game-conserved
                    (not
                      (and
                        (in_motion ?j)
                        (not
                          (adjacent block ?r)
                        )
                      )
                    )
                  )
                )
              )
            )
            (game-conserved
              (not
                (and
                  (exists (?h - hexagonal_bin)
                    (and
                      (in_motion agent)
                      (in_motion agent ?j)
                    )
                  )
                  (not
                    (agent_holds ?y)
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
    (preference preference1
      (then
        (hold (on ?xxx) )
        (once (not (in_motion agent ?xxx) ) )
        (once (not (not (agent_holds ?xxx ?xxx) ) ) )
        (once (adjacent upright) )
      )
    )
  )
)
(:terminal
  (or
    (or
      (>= 1 (count preference1:dodgeball) )
      (or
        (not
          (and
            (or
              (not
                (>= (+ 1 4 2 )
                  (+ 6 (* (* (count preference1:block) 3 )
                      (count-same-positions preference1:white:golfball)
                      (or
                        2
                        (count-once-per-objects preference1:beachball:pink:green)
                        (count-once-per-objects preference1:pink)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (and
          (and
            (>= (< 7 5 )
              (+ 10 (* (count-once-per-objects preference1:purple) 3 )
              )
            )
            (>= 1 (count preference1:book:beachball) )
            (>= (count-once preference1:green:tall_cylindrical_block) (>= (count-once-per-objects preference1:beachball) (total-score) )
            )
          )
          (> (external-forall-maximize (>= (external-forall-maximize (+ (- (count preference1:hexagonal_bin) )
                    (* (+ (* 4 (count-once-per-objects preference1:yellow:dodgeball) )
                        (count preference1:doggie_bed)
                      )
                      (count preference1:yellow_cube_block:pink)
                    )
                  )
                )
                (count preference1:dodgeball)
              )
            )
            (+ (* (+ (* (* (+ (count preference1:beachball) )
                      (count-once-per-objects preference1:basketball)
                      (count preference1:dodgeball)
                    )
                    (* 3 )
                  )
                  3
                )
                (count-shortest preference1:red)
                (total-score)
                (count-once-per-objects preference1:pink_dodgeball:golfball)
                (count preference1:green)
                (total-score)
                (total-time)
                (count-once-per-objects preference1:doggie_bed)
                (count preference1:blue_dodgeball)
                (* (+ (count-once preference1:beachball:wall) 30 (* 4 (total-time) (count-shortest preference1:basketball:orange) )
                  )
                  (count preference1:hexagonal_bin)
                )
                3
                (* 6 (count preference1:tall_cylindrical_block) )
              )
              12
            )
          )
          (< (* (+ 3 (count preference1:red:hexagonal_bin) )
              (+ (count preference1:blue:doggie_bed) (* (* (* 20 2 )
                    (* (* (* 4 (count-once-per-objects preference1:dodgeball:basketball) (external-forall-maximize (count preference1:pink_dodgeball) ) )
                        8
                      )
                      (+ 10 (* (not 10 ) (* (count-once-per-external-objects preference1:dodgeball:beachball) 30 )
                        )
                      )
                      1
                    )
                    20
                    (count-once preference1:purple:hexagonal_bin)
                    (* (count preference1:basketball) 4 )
                    (count preference1:red_pyramid_block:beachball:green)
                  )
                  (count-once-per-objects preference1:dodgeball)
                )
              )
            )
            (* (total-time) 5 )
          )
        )
      )
      (> (count-once-per-objects preference1:golfball) 2 )
    )
    (>= (count preference1:dodgeball) (* 3 (+ 2 (* 2 10 )
        )
        (count-overlapping preference1:purple:green)
        (count-once-per-objects preference1:yellow)
        (* (count preference1:rug) 1 (+ (- 9 )
            (* 300 (external-forall-minimize (/ 5 (* (total-time) 2 )
                )
              )
            )
          )
        )
        6
        (count preference1:golfball)
        (+ (count-once-per-objects preference1:beachball) (* (* (* (count preference1:dodgeball) (* (* 3 (* 20 (* 7 (count-once-per-objects preference1:purple) (count preference1:yellow) )
                    )
                  )
                  6
                )
              )
              (count preference1:green)
            )
            12
          )
          (count preference1:beachball)
        )
        (count preference1:pink_dodgeball:dodgeball:yellow)
      )
    )
  )
)
(:scoring
  60
)
)


(define (game game-id-99) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?n - ball)
      (exists (?t - chair)
        (game-conserved
          (in ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - hexagonal_bin)
      (and
        (preference preference1
          (at-end
            (and
              (agent_holds south_west_corner)
              (and
                (agent_holds ?i ?i)
                (agent_holds front ?i)
              )
            )
          )
        )
        (preference preference2
          (then
            (hold (in_motion ?i) )
            (once (or (in_motion ?i ?i) ) )
            (once (in_motion agent ?i) )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count preference2:pink) (* (>= (* 6 (total-score) )
          (- (total-time) )
        )
        (/
          1
          (count preference1:purple)
        )
      )
    )
    (and
      (>= (<= 4 (count preference1:side_table:bed) )
        5
      )
      (and
        (>= (* (count-once preference2:yellow:basketball) 6 )
          (count preference1:block)
        )
        (>= (+ (external-forall-maximize (- (count-once preference1:dodgeball:pink_dodgeball) (* (count preference2:pink) (* 2 (* (* 3 (not (* (count-once-per-objects preference2:rug:beachball) (count-once-per-objects preference1:yellow:hexagonal_bin) )
                        )
                      )
                      (total-score)
                      (count preference1)
                    )
                    5
                  )
                  4
                  12
                  (* 6 (count preference2:dodgeball:blue_pyramid_block:beachball) 50 (* (total-time) 10 (* 15 (+ 5 2 )
                      )
                      (count preference2:golfball)
                    )
                  )
                  (* (count-shortest preference2:hexagonal_bin) (count-once-per-external-objects preference1:dodgeball) )
                  3
                  (- (count preference2:dodgeball) )
                  (count preference2:blue_pyramid_block)
                )
              )
            )
            (count preference1:orange)
            8
          )
          (* (or (count preference2:yellow) (count preference2:basketball) (* (* (count-once-per-objects preference2:green:golfball) (count-same-positions preference1:blue_dodgeball:golfball) (and (* (count-once-per-objects preference1:book) 6 )
                    (count-increasing-measure preference2:wall:beachball)
                    3
                  )
                  10
                )
                (count preference2:doggie_bed)
                (* (count preference1:golfball:red) (count-once preference1:hexagonal_bin:basketball:dodgeball) )
                (count-once-per-objects preference2:yellow:dodgeball)
              )
            )
            (count preference1:hexagonal_bin:blue_dodgeball:dodgeball)
          )
        )
      )
    )
  )
)
(:scoring
  (count preference2:green)
)
)

