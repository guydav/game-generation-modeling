
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
    (forall (?z - hexagonal_bin ?z - beachball)
      (and
        (preference preference1
          (at-end
            (and
              (touch ?z)
              (and
                (faces ?z)
                (in_motion ?z)
              )
              (in_motion ?z)
              (< (distance ?z ?z) (distance ?z 7))
              (agent_holds ?z)
              (on ?z ?z)
            )
          )
        )
        (preference preference2
          (then
            (once (not (and (not (and (agent_holds ?z) (agent_holds ?z) (and (not (not (touch ?z ?z) ) ) (exists (?u - ball ?l - shelf) (and (touch pink ?l) (not (and (in_motion ?z) (not (and (on ?l ?l) (and (exists (?s - triangular_ramp) (agent_holds ?s) ) (rug_color_under ?l) (< (distance front ?z) 1) (exists (?e - dodgeball) (agent_holds ?l agent) ) ) ) ) ) ) ) ) ) (agent_holds ?z ?z) ) ) (or (and (toggled_on ?z) (in ?z) ) (agent_holds ?z) ) ) ) )
            (once (> (distance room_center ?z) 2) )
            (once (between ?z) )
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
    (exists (?s - (either ball yellow_cube_block) ?s - (either book cylindrical_block) ?r - ball)
      (exists (?x - block ?g ?c - (either basketball dodgeball bridge_block))
        (forall (?x - doggie_bed)
          (game-conserved
            (agent_holds ?r)
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
    (forall (?g - (either blue_cube_block tall_cylindrical_block golfball))
      (and
        (preference preference1
          (then
            (hold (equal_z_position ?g ?g) )
            (once (in_motion ?g desk) )
            (once (and (in_motion ?g) (and (and (and (agent_holds ?g) (and (agent_holds ?g) (in_motion floor) ) ) (not (on ?g desk) ) ) (adjacent ?g) (not (in_motion ?g ?g) ) ) ) )
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
  (exists (?h - block)
    (forall (?d - wall)
      (game-optional
        (= 2 0 8)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b ?r - dodgeball)
        (then
          (once (in_motion ?b ?b) )
          (once (not (exists (?f - block ?a - hexagonal_bin) (on ?b) ) ) )
          (hold (and (on agent) (touch ?b ?r) ) )
        )
      )
    )
    (preference preference2
      (exists (?d - game_object)
        (at-end
          (not
            (not
              (not
                (agent_holds ?d)
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
  (exists (?p - game_object ?u - bridge_block)
    (and
      (and
        (and
          (forall (?w - game_object ?y - hexagonal_bin)
            (game-conserved
              (< 1 1)
            )
          )
          (forall (?n - hexagonal_bin)
            (and
              (game-conserved
                (in ?n)
              )
            )
          )
        )
        (forall (?c ?v - hexagonal_bin)
          (and
            (game-optional
              (and
                (not
                  (not
                    (and
                      (on ?u ?v)
                      (and
                        (not
                          (in_motion ?v)
                        )
                        (not
                          (and
                            (agent_holds ?u agent)
                            (and
                              (not
                                (and
                                  (adjacent desk)
                                  (not
                                    (in_motion ?v ?v)
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
                (in_motion ?v)
              )
            )
            (and
              (forall (?r ?p ?n ?h ?d - yellow_cube_block ?g - red_dodgeball)
                (exists (?d - ball)
                  (game-conserved
                    (in_motion agent)
                  )
                )
              )
            )
            (game-conserved
              (in ?u ?c)
            )
          )
        )
        (exists (?h - hexagonal_bin ?e - wall ?n - hexagonal_bin)
          (exists (?j - rug ?l - dodgeball ?k - game_object ?l - (either lamp curved_wooden_ramp))
            (game-conserved
              (not
                (and
                  (in ?n)
                  (on ?n upside_down)
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
    (forall (?z - hexagonal_bin ?q - dodgeball ?h - cylindrical_block)
      (and
        (preference preference1
          (exists (?e - (either dodgeball golfball cube_block))
            (then
              (once (on ?e ?h) )
              (once (in_motion agent ?h ?h) )
              (once (agent_holds ?e) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?h - wall)
        (then
          (once (and (touch agent) (and (or (and (not (and (same_type ?h ?h) (and (exists (?z - red_dodgeball ?d - beachball ?z - (either key_chain)) (agent_holds rug ?h ?h) ) (adjacent ?h) ) (not (agent_holds top_shelf) ) ) ) (and (adjacent ?h ?h) (< 1 0.5) (in_motion front_left_corner) (and (touch ?h ?h) (adjacent ?h ?h) ) ) ) ) (and (in_motion south_west_corner) (broken rug ?h) ) (in_motion rug) ) ) )
          (once (and (in ?h agent) (and (and (in_motion ?h) (< 1 1) (and (on agent) (not (agent_holds ?h) ) ) (and (on ?h agent) (and (not (on bed ?h) ) (not (agent_holds ?h bed) ) (agent_holds ?h) (and (adjacent ?h ?h) (on agent ?h) ) ) ) ) (in rug) ) ) )
          (once (or (and (adjacent ?h ?h) (in_motion agent ?h) ) (in_motion ?h) ) )
        )
      )
    )
    (preference preference3
      (exists (?r - color ?s - dodgeball)
        (then
          (once-measure (and (and (and (in_motion bed) (and (in_motion agent ?s) (agent_holds ?s) ) (in_motion ?s ?s) (in_motion ?s ?s) (and (in ?s) (or (not (not (agent_holds ?s ?s) ) ) (in ?s) ) (agent_holds ?s ?s) ) (in ?s) ) (not (and (not (on ?s) ) (in ?s ?s) ) ) ) (in ?s top_drawer) ) (building_size 4 ?s) )
          (once (agent_holds ?s) )
          (hold-while (touch ?s ?s) (not (> 5 1) ) )
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
    (forall (?j - hexagonal_bin)
      (and
        (preference preference1
          (then
            (any)
            (once (touch desk ?j) )
            (once (not (> 1 (distance agent ?j)) ) )
          )
        )
      )
    )
    (preference preference2
      (exists (?r - curved_wooden_ramp ?a - cube_block)
        (exists (?m - hexagonal_bin)
          (then
            (hold (forall (?u - (either dodgeball)) (agent_holds ?u ?a) ) )
            (hold (and (and (and (not (not (in ?a) ) ) (and (not (agent_holds ?a ?a) ) (in ?m) ) ) (and (on tan) (exists (?f - hexagonal_bin) (not (exists (?u - shelf) (in_motion ?f) ) ) ) ) ) (not (agent_holds pink ?m) ) (and (and (and (on upright ?m) (and (same_type ?a agent) (on ?m brown) ) ) (= (distance ?a ?a) 1 (distance ?m 10)) (and (on ?a) (and (on ?a ?m) (agent_holds ?m) ) ) ) (not (agent_holds ?m) ) ) ) )
            (once (agent_holds ?m) )
          )
        )
      )
    )
    (forall (?i - dodgeball)
      (and
        (preference preference3
          (exists (?z - hexagonal_bin)
            (exists (?a - building)
              (at-end
                (not
                  (and
                    (not
                      (not
                        (not
                          (not
                            (and
                              (not
                                (toggled_on ?a)
                              )
                              (agent_holds ?a)
                              (in_motion ?z)
                            )
                          )
                        )
                      )
                    )
                    (< (distance ?a ?a) 1)
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
      (exists (?y - dodgeball ?l - hexagonal_bin)
        (then
          (hold (same_object ?l) )
          (hold (and (and (in ?l) (touch ?l) (not (on ?l) ) ) (exists (?n - teddy_bear) (adjacent ?l) ) ) )
          (once (in_motion ?l ?l) )
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
      (exists (?h - pillow)
        (then
          (once (agent_holds ?h) )
          (once (> 1 (distance ?h)) )
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
    (forall (?c - dodgeball)
      (and
        (preference preference1
          (exists (?i - watch)
            (exists (?g - cube_block)
              (forall (?t - block)
                (then
                  (once (and (agent_holds ?t) (not (equal_z_position ?i ?g) ) ) )
                  (once (= 1 1) )
                  (once (not (in_motion ?t bed) ) )
                )
              )
            )
          )
        )
        (preference preference2
          (then
            (once (on ?c ?c) )
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
        (hold-while (not (> (distance 1 2) 7) ) (and (not (not (and (< 1 1) (agent_holds ?xxx ?xxx ?xxx) ) ) ) (and (not (and (and (on ?xxx) (not (on ?xxx) ) ) (not (= 1 (x_position ?xxx ?xxx)) ) ) ) (not (and (in_motion ?xxx) (or (agent_holds ?xxx) (and (agent_holds floor ?xxx) (agent_holds ?xxx sideways) ) ) ) ) ) ) (exists (?d - game_object) (< (distance room_center 4) (distance ?d desk)) ) (on ?xxx) )
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
      (exists (?y - ball)
        (exists (?p - (either cube_block hexagonal_bin) ?m - beachball ?x - hexagonal_bin)
          (game-conserved
            (adjacent ?x green_golfball)
          )
        )
      )
    )
    (exists (?s - rug)
      (game-conserved
        (and
          (and
            (and
              (in_motion ?s ?s)
              (touch ?s)
            )
            (agent_holds ?s)
          )
          (in_motion ?s ?s)
          (on ?s)
        )
      )
    )
    (exists (?z - doggie_bed ?z - (either yellow_cube_block wall) ?w - hexagonal_bin)
      (forall (?o - dodgeball)
        (game-optional
          (and
            (and
              (in_motion ?o ?o)
              (< (distance ?w ?o) 2)
            )
            (in ?w)
            (< 0 (distance ?w ?o))
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - (either pyramid_block cube_block))
        (at-end
          (is_setup_object bed ?n)
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
      (exists (?n - building)
        (exists (?v - block ?m - golfball)
          (exists (?r - game_object)
            (exists (?o - doggie_bed)
              (exists (?x - triangular_ramp ?i - game_object)
                (exists (?a ?p - teddy_bear ?b - (either yellow_cube_block laptop) ?j - hexagonal_bin ?z - (either tall_cylindrical_block hexagonal_bin pyramid_block))
                  (exists (?d - hexagonal_bin)
                    (then
                      (once (in ?d) )
                      (hold-while (not (in_motion ?o pink) ) (touch ?r) (and (opposite ?o) ) )
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
        (once (not (exists (?x - doggie_bed) (not (and (and (and (and (adjacent ?x) (in_motion ?x) ) (in ?x agent) ) (agent_holds agent ?x) ) (not (< (distance side_table) 6) ) ) ) ) ) )
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
        (hold-while (in_motion ?xxx ?xxx) (and (same_color ?xxx) (exists (?m - hexagonal_bin) (and (not (adjacent_side ?m ?m) ) (in_motion ?m agent) ) ) ) )
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
              (exists (?a ?z - game_object)
                (agent_holds ?z ?z)
              )
            )
          )
          (agent_holds ?xxx)
        )
      )
      (not
        (exists (?i - chair ?k - hexagonal_bin ?d - ball)
          (and
            (on ?d ?d)
            (agent_holds ?d ?d)
            (agent_holds ?d)
            (and
              (and
                (adjacent ?d desk)
                (and
                  (agent_holds ?d)
                  (adjacent ?d)
                )
                (and
                  (and
                    (agent_holds ?d ?d)
                    (or
                      (not
                        (on sideways ?d)
                      )
                      (in ?d)
                    )
                  )
                  (= 9 2 7)
                )
              )
              (not
                (not
                  (agent_holds ?d)
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
      (exists (?j ?f ?h ?o - hexagonal_bin)
        (then
          (once (in_motion ?f ?h) )
          (once (agent_holds ?o) )
          (hold (adjacent ?f) )
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
      (exists (?q - dodgeball ?s - (either block blue_cube_block))
        (at-end
          (in ?s)
        )
      )
    )
    (preference preference2
      (exists (?w - hexagonal_bin)
        (then
          (hold-to-end (touch ?w) )
          (hold (not (rug_color_under ?w ?w) ) )
          (hold-while (not (and (not (and (not (and (agent_holds ?w) (not (agent_holds ?w) ) ) ) (adjacent floor) ) ) (on ?w) ) ) (and (and (same_object ?w) (touch ?w) ) (not (agent_holds ?w) ) ) )
        )
      )
    )
    (forall (?i - pillow)
      (and
        (preference preference3
          (then
            (once (and (in_motion ?i) (and (in_motion ?i ?i) (agent_holds ?i ?i) ) ) )
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
    (exists (?d - (either cd dodgeball))
      (and
        (exists (?r - hexagonal_bin)
          (forall (?x - block ?u - (either blue_cube_block pyramid_block))
            (game-conserved
              (in_motion ?d)
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
    (forall (?b - hexagonal_bin)
      (and
        (preference preference2
          (exists (?w - (either book wall))
            (at-end
              (in_motion ?b)
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
      (exists (?w - hexagonal_bin)
        (then
          (once (adjacent_side side_table) )
          (once (not (agent_holds ?w) ) )
          (forall-sequence (?z - cube_block)
            (then
              (once (and (agent_holds ?z ?z) (and (in_motion ?w ?w) ) ) )
              (hold-while (not (in_motion agent) ) (adjacent ?w) (on agent ?w) (and (< 1 1) (not (>= 1 1) ) ) )
              (once (on ?z ?w) )
            )
          )
        )
      )
    )
    (forall (?w - cube_block)
      (and
        (preference preference2
          (exists (?o - (either pyramid_block dodgeball beachball))
            (at-end
              (in_motion bed ?w)
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
      (exists (?y - game_object ?y - ball)
        (then
          (hold (and (< 4 1) (adjacent ?y) ) )
          (hold-while (in_motion ?y) (not (and (in agent agent) ) ) )
          (once (in_motion bed ?y) )
          (once (in_motion agent ?y) )
        )
      )
    )
    (preference preference2
      (exists (?c ?m ?f ?j ?n ?k - (either yellow_cube_block bridge_block pyramid_block) ?k - dodgeball ?o - drawer)
        (exists (?j - game_object)
          (then
            (hold-to-end (agent_holds ?j ?o) )
            (once (on ?j) )
            (hold-while (not (not (in_motion ?o) ) ) (agent_holds ?o ?o) )
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
  (exists (?n - pyramid_block ?v - game_object)
    (game-optional
      (in_motion agent ?v)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (agent_holds ?xxx) (or (not (adjacent ?xxx ?xxx) ) (and (in_motion ?xxx) (not (agent_holds back) ) ) ) ) )
        (once (on ?xxx ?xxx) )
        (forall-sequence (?h - building)
          (then
            (once (not (adjacent pink_dodgeball) ) )
            (once-measure (and (between ?h ?h) ) (distance desk ?h) )
            (once (in_motion ?h) )
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
  (forall (?r - beachball)
    (game-conserved
      (agent_holds ?r ?r)
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
      (exists (?z - hexagonal_bin)
        (then
          (once (> 1 (distance )) )
          (once (adjacent ?z) )
          (once (agent_holds ?z) )
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
    (exists (?d - cube_block)
      (or
        (game-conserved
          (and
            (not
              (on ?d ?d ?d)
            )
            (and
              (not
                (agent_holds ?d)
              )
              (agent_holds ?d ?d)
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
      (exists (?l - chair ?k - (either main_light_switch))
        (then
          (any)
          (hold (on floor ?k) )
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
      (exists (?t - game_object)
        (in_motion ?t ?t ?t)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - beachball ?w - dodgeball)
        (exists (?m - building)
          (exists (?u - curved_wooden_ramp ?h - desk_shelf)
            (exists (?u - cube_block)
              (then
                (once (same_object ?u) )
                (once (agent_holds ?m) )
                (once (< 1 1) )
                (once (and (adjacent ?u) (on ?u ?h) (not (not (in_motion ?m) ) ) (or (not (and (on ?u) (and (game_over ?m ?w) (on ?m brown) ) (agent_holds ?h) (and (and (and (agent_holds front) (agent_holds ?h) (above ?m) ) (agent_holds ?h ?m) (and (not (or (< (distance front_left_corner ?h) 7) (agent_holds desk ?h) ) ) (or (and (forall (?y - curved_wooden_ramp) (not (in ?y ?m) ) ) (touch ?m ?w) ) (adjacent ?m ?h) ) ) ) (not (agent_holds floor ?m) ) ) (= (distance ) (distance ?u ?u) (distance ?h ?h)) (in ?m) (in upright floor ?h ?h) (faces ?w) ) ) (not (same_object agent) ) ) ) )
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
    (forall (?e - dodgeball)
      (and
        (preference preference3
          (at-end
            (in_motion ?e agent)
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
  (exists (?f - (either key_chain laptop) ?s - tall_cylindrical_block ?c - color)
    (not
      (and
        (game-optional
          (< (distance ?c) 0)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - dodgeball)
      (and
        (preference preference1
          (exists (?x - (either book blue_cube_block))
            (then
              (once (on ?x ?x) )
              (hold (not (in_motion ?x) ) )
              (once (not (in_motion ?a ?x) ) )
            )
          )
        )
      )
    )
    (forall (?t - dodgeball)
      (and
        (preference preference2
          (at-end
            (agent_holds ?t)
          )
        )
      )
    )
    (preference preference3
      (then
        (once (same_color ?xxx agent) )
        (hold-while (not (agent_holds ?xxx) ) (in ?xxx ?xxx) (exists (?d - (either cube_block)) (agent_holds ?d) ) )
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
  (forall (?e - hexagonal_bin)
    (forall (?l - hexagonal_bin)
      (exists (?h - watch)
        (or
          (forall (?w - cube_block ?q - dodgeball)
            (forall (?o - hexagonal_bin)
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
)
(:constraints
  (and
    (preference preference1
      (exists (?x - doggie_bed)
        (exists (?d - dodgeball)
          (exists (?k - (either cellphone dodgeball) ?o - (either cylindrical_block flat_block pyramid_block))
            (exists (?f - (either dodgeball book) ?v - (either cd pyramid_block doggie_bed))
              (exists (?q - wall)
                (exists (?n - cube_block ?l - dodgeball)
                  (exists (?e - hexagonal_bin ?h - chair)
                    (exists (?w ?n - cube_block ?r - (either laptop pillow blue_cube_block) ?f - dodgeball ?p - wall)
                      (at-end
                        (in_motion ?v)
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
      (exists (?h - dodgeball)
        (then
          (once (and (not (and (not (not (adjacent ?h) ) ) (adjacent pillow) ) ) (same_type ?h) ) )
          (once (forall (?q - color) (exists (?c - curved_wooden_ramp) (agent_holds ?h) ) ) )
          (once (adjacent ?h desk) )
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
  (forall (?d - hexagonal_bin)
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
      (forall (?g ?j ?r ?w - game_object ?f - doggie_bed ?e - game_object)
        (then
          (once (not (on rug) ) )
          (once (agent_holds sideways) )
          (hold-while (and (agent_holds ?e) (on ?e ?e) (exists (?g ?u ?i ?v - (either flat_block mug alarm_clock)) (on ?v ?u) ) ) (and (on ?e ?e) ) )
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
    (forall (?n - drawer ?f - hexagonal_bin ?j - building)
      (and
        (preference preference1
          (then
            (hold (agent_holds agent ?j) )
            (hold (in_motion ?j) )
            (hold (in ?j) )
          )
        )
        (preference preference2
          (then
            (once (game_over ?j) )
            (hold (object_orientation ?j) )
            (once (not (not (not (and (and (not (and (in_motion ?j) (not (in_motion desk ?j) ) (exists (?o - curved_wooden_ramp) (and (and (and (exists (?a - hexagonal_bin) (agent_holds ?o pink_dodgeball) ) (in_motion bed) ) ) (and (and (> 5 1) (and (and (adjacent ?o ?j) (not (on ?o) ) ) (not (agent_holds ?o) ) (on ?o) ) (in ?o ?j) ) (in_motion ?o ?j) ) ) ) (agent_holds ?j) (and (agent_holds ?j) (not (and (agent_holds pink desk) (in ?j ?j) ) ) (not (and (in_motion ?j) ) ) ) (in ?j ?j) (agent_holds ?j ?j) ) ) (in_motion door) ) (not (in_motion agent ?j) ) ) ) ) ) )
          )
        )
      )
    )
    (preference preference3
      (exists (?v - (either laptop pen cellphone))
        (exists (?p - hexagonal_bin ?l - game_object ?c - pyramid_block)
          (then
            (once (>= (distance ?c ?v) (distance ?c side_table)) )
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
  (forall (?m - shelf)
    (exists (?g - dodgeball)
      (game-conserved
        (and
          (and
            (not
              (touch ?m)
            )
            (and
              (on ?g)
              (in_motion ?g)
            )
            (< (distance agent ?g) 1)
          )
          (in_motion ?g)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - dodgeball)
      (and
        (preference preference1
          (then
            (hold (on ?d bed) )
            (once (exists (?z - hexagonal_bin) (in_motion ?d) ) )
            (forall-sequence (?b - game_object)
              (then
                (once (and (>= (distance 7 desk) (distance ?b ?b)) (on ?b) ) )
                (once-measure (in_motion agent) (distance back room_center) )
                (hold (not (on ?d ?b) ) )
              )
            )
            (once (on ?d) )
          )
        )
        (preference preference2
          (then
            (once (agent_holds ?d ?d) )
            (hold (not (not (not (and (in ?d ?d) (and (and (adjacent pink_dodgeball ?d) (not (not (adjacent pink) ) ) ) (game_start ?d) (in_motion ?d floor) (agent_holds ?d) ) ) ) ) ) )
            (once (in_motion ?d ?d) )
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
    (forall (?t - ball ?x - teddy_bear)
      (game-conserved
        (exists (?q - hexagonal_bin)
          (touch ?q ?q ?x)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?l - game_object)
      (and
        (preference preference1
          (exists (?f - wall)
            (then
              (once (in_motion ?f ?f) )
              (once (in_motion ?f ?l) )
              (once (not (agent_holds top_shelf ?f) ) )
            )
          )
        )
      )
    )
    (forall (?d - (either game_object lamp cellphone) ?q - building)
      (and
        (preference preference2
          (exists (?d - dodgeball ?n - ball)
            (exists (?a - (either blue_cube_block credit_card laptop) ?p - game_object)
              (exists (?j - dodgeball)
                (exists (?o - beachball ?f - ball ?i - cube_block)
                  (then
                    (once (not (in_motion ?j) ) )
                    (hold (agent_holds ?i) )
                    (once (agent_holds ?q ?n) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?i - hexagonal_bin)
        (then
          (hold (or (not (in ?i) ) (above ?i ?i) (exists (?f - (either dodgeball desktop)) (in ?f) ) (in_motion ?i ?i) ) )
          (once (adjacent agent ?i) )
          (once (is_setup_object ?i) )
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
  (forall (?m - game_object)
    (exists (?f - desk_shelf)
      (game-optional
        (in_motion ?m ?m)
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
    (forall (?q - (either tall_cylindrical_block yellow_cube_block) ?n - dodgeball)
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
    (forall (?h - block)
      (and
        (preference preference1
          (exists (?z - hexagonal_bin ?e - cube_block)
            (exists (?i - teddy_bear)
              (forall (?g - dodgeball)
                (exists (?c - wall)
                  (exists (?w - triangular_ramp)
                    (then
                      (once (agent_holds ?h) )
                      (once (on pillow) )
                      (once (on ?w ?w) )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
    (forall (?u - hexagonal_bin)
      (and
        (preference preference2
          (then
            (once (> (distance ?u ?u) 0) )
            (hold (in ?u) )
            (hold-while (exists (?r - building) (adjacent ?r) ) (agent_holds ?u ?u) )
          )
        )
        (preference preference3
          (at-end
            (> (distance ?u 9) 2)
          )
        )
        (preference preference4
          (exists (?v - chair)
            (exists (?e - ball)
              (forall (?p - hexagonal_bin ?x - doggie_bed)
                (exists (?j - chair)
                  (then
                    (once (agent_holds ?v ?u) )
                    (once (exists (?z - chair ?s - block) (in_motion ?j) ) )
                    (hold (and (touch ?x) (< 9 2) (or (agent_holds ?j ?v) (and (< (distance ?v ?e) (distance_side ?e ?e)) (agent_holds ?e) ) ) (in_motion ?e) (adjacent ?v) (agent_holds ?v ?v) (agent_holds ?v ?x) (not (< (distance ?u ?x) 1) ) (and (not (and (is_setup_object ?e) (touch agent) ) ) (agent_holds ?v) ) (and (and (and (touch ?e) (agent_holds pink) ) (in_motion ?u) ) (not (and (in pillow ?e) (agent_holds ?e ?x) (agent_holds ?v) ) ) (in_motion ?u) ) (in_motion ?v ?u) (agent_holds ?x sideways desk) ) )
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
  (forall (?s - game_object ?v - chair ?s - block)
    (game-optional
      (adjacent_side ?s ?s)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (on desk ?xxx) )
        (forall-sequence (?k - dodgeball)
          (then
            (hold (not (agent_holds bed ?k) ) )
            (hold (agent_holds ?k) )
            (hold-while (exists (?p - hexagonal_bin) (in ?k) ) (in_motion ?k) )
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
        (exists (?s - (either flat_block desktop))
          (exists (?d - game_object)
            (game-conserved
              (and
                (not
                  (equal_z_position ?s ?s)
                )
                (in_motion ?d)
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
      (exists (?u - (either golfball dodgeball))
        (exists (?q ?n - chair)
          (then
            (any)
            (once (agent_holds ?u ?n) )
            (once (not (in_motion ?q ?q) ) )
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
  (forall (?f - cube_block)
    (game-conserved
      (in ?f)
    )
  )
)
(:constraints
  (and
    (forall (?w - hexagonal_bin)
      (and
        (preference preference1
          (then
            (once (in ?w) )
            (hold (in ?w ?w) )
            (once-measure (in_motion ?w ?w) (distance ?w front_left_corner) )
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
      (exists (?l - hexagonal_bin)
        (then
          (once-measure (object_orientation ?l) (distance 4 ?l ?l) )
          (once (touch ?l ?l) )
          (hold-while (in_motion bed agent) (not (not (equal_z_position ?l) ) ) (and (not (and (in ?l ?l) (agent_holds ?l) ) ) (rug_color_under ?l ?l) ) )
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
  (exists (?f - hexagonal_bin)
    (game-conserved
      (not
        (on desktop front)
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - (either cd teddy_bear blue_cube_block))
      (and
        (preference preference1
          (forall (?b - ball)
            (exists (?u - hexagonal_bin)
              (exists (?h - cube_block)
                (then
                  (hold (and (in_motion ?a) (agent_holds ?a) ) )
                  (once (and (and (on ?a) (in_motion ?b) ) (not (on ?b) ) ) )
                  (once (and (agent_holds ?a ?a) (on ?u ?u) ) )
                )
              )
            )
          )
        )
        (preference preference2
          (exists (?k - ball ?y - dodgeball)
            (then
              (once (in ?a) )
              (once (agent_holds ?a ?a pink) )
              (once (touch ?y) )
            )
          )
        )
      )
    )
    (preference preference3
      (forall (?k - hexagonal_bin ?c - dodgeball)
        (exists (?s - dodgeball)
          (then
            (hold (and (and (agent_holds ?s) (agent_holds ?c) ) (and (not (same_color ?c ?s) ) (or (in_motion ?s ?s) (in_motion ?c) ) ) ) )
            (once (not (in_motion ?s ?c) ) )
            (hold (and (on ?c) (in_motion ?c) ) )
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
    (exists (?r - block)
      (game-conserved
        (on ?r)
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
    (forall (?d - doggie_bed ?x - (either pencil pencil) ?v - cube_block)
      (game-conserved
        (agent_holds brown)
      )
    )
  )
)
(:constraints
  (and
    (forall (?m - dodgeball ?o - cube_block)
      (and
        (preference preference1
          (then
            (once (not (not (and (game_start ?o) (in_motion ?o ?o) ) ) ) )
            (hold-for 6 (equal_z_position ?o) )
            (once (in_motion left ?o) )
          )
        )
        (preference preference2
          (exists (?u - (either floor dodgeball))
            (exists (?b - dodgeball ?v - dodgeball)
              (at-end
                (in block ?o)
              )
            )
          )
        )
        (preference preference3
          (exists (?e - dodgeball ?s ?d ?b - hexagonal_bin ?r - dodgeball)
            (then
              (once (on ?o) )
              (hold-while (in_motion ?o agent) (agent_holds ?r) )
            )
          )
        )
      )
    )
    (forall (?b - doggie_bed)
      (and
        (preference preference4
          (exists (?k - hexagonal_bin)
            (then
              (hold (in ?b) )
              (once (not (and (adjacent ?b ?b ?b) (in_motion ?b) (< 2 1) ) ) )
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
      (forall (?c - (either doggie_bed blue_cube_block))
        (in_motion ?c)
      )
      (agent_holds agent)
    )
  )
)
(:constraints
  (and
    (forall (?e - hexagonal_bin)
      (and
        (preference preference1
          (then
            (hold (adjacent ?e ?e) )
            (once (and (agent_holds green ?e) (< 1 9) ) )
            (once (same_color ?e ?e) )
            (hold (not (and (in_motion ?e) (not (not (adjacent rug) ) ) ) ) )
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
  (forall (?g - doggie_bed)
    (game-conserved
      (on ?g)
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
    (forall (?l - ball ?j - shelf)
      (and
        (preference preference1
          (at-end
            (and
              (not
                (in ?j)
              )
              (agent_holds ?j)
            )
          )
        )
        (preference preference2
          (exists (?g - hexagonal_bin)
            (exists (?s - block)
              (exists (?o - hexagonal_bin)
                (exists (?r - dodgeball)
                  (exists (?e - ball ?p - block ?u - hexagonal_bin ?p - chair)
                    (exists (?v - game_object ?f - (either top_drawer cube_block bridge_block))
                      (then
                        (once (agent_holds ?s ?r) )
                        (once (not (on green_golfball) ) )
                        (hold (not (and (in ?j) (not (not (not (in_motion ?g ?r) ) ) ) ) ) )
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
      (exists (?h - teddy_bear ?k - (either dodgeball golfball))
        (then
          (once (on ?k ?k) )
          (once (and (and (in_motion ?k) (touch ?k) ) (and (rug_color_under ?k ?k) (agent_holds ) ) ) )
          (once (in_motion rug ?k) )
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
      (exists (?p - wall)
        (exists (?k - (either chair triangular_ramp))
          (then
            (hold (on ?k) )
            (once (and (and (in_motion desk) (same_color ?k) ) (and (in ?k ?k) (on ?p ?p) (not (not (not (not (in_motion ?k ?k) ) ) ) ) (in_motion ?k) ) ) )
            (once (agent_holds ?p ?k) )
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
        (exists (?w - ball)
          (game-conserved
            (in ?w)
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
          (forall (?h - hexagonal_bin ?h - wall)
            (game-conserved
              (not
                (< (distance ?h 9) (distance agent ?h))
              )
            )
          )
          (and
            (exists (?m - dodgeball)
              (game-conserved
                (on ?m ?m)
              )
            )
          )
          (forall (?m - ball)
            (and
              (game-optional
                (agent_holds ?m ?m)
              )
            )
          )
          (forall (?l - dodgeball)
            (exists (?u - block)
              (and
                (and
                  (forall (?e - hexagonal_bin ?q - dodgeball)
                    (game-conserved
                      (in_motion upright)
                    )
                  )
                  (and
                    (and
                      (game-conserved
                        (agent_holds bed ?l)
                      )
                      (game-conserved
                        (agent_holds ?u ?l)
                      )
                    )
                  )
                )
              )
            )
          )
          (exists (?q - flat_block ?r - (either main_light_switch cube_block) ?s - curved_wooden_ramp)
            (forall (?v - (either dodgeball bridge_block) ?y - pyramid_block)
              (game-conserved
                (not
                  (not
                    (exists (?z - hexagonal_bin)
                      (not
                        (same_color ?z)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (forall (?x ?b ?t ?l ?a ?f - hexagonal_bin ?t - hexagonal_bin)
          (exists (?i - ball)
            (and
              (forall (?o - (either pillow book))
                (and
                  (forall (?b - (either key_chain yellow))
                    (game-optional
                      (on ?b ?t)
                    )
                  )
                )
              )
              (and
                (game-optional
                  (touch ?t ?t)
                )
                (not
                  (and
                    (game-conserved
                      (and
                        (and
                          (not
                            (agent_holds ?t ?i)
                          )
                          (adjacent ?i)
                          (not
                            (same_color ?i)
                          )
                          (not
                            (in_motion ?t)
                          )
                          (and
                            (not
                              (not
                                (not
                                  (same_color ?i)
                                )
                              )
                            )
                            (agent_holds ?i)
                          )
                          (in_motion ?i)
                        )
                        (same_color ?i)
                      )
                    )
                    (exists (?c - wall ?n - triangular_ramp ?m - (either bridge_block doggie_bed))
                      (and
                        (game-optional
                          (agent_holds ?m ?i)
                        )
                        (forall (?v - chair)
                          (game-conserved
                            (game_start ?i)
                          )
                        )
                      )
                    )
                    (game-conserved
                      (in_motion ?i ?t)
                    )
                  )
                )
                (game-conserved
                  (not
                    (in_motion bridge_block ?i)
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
      (exists (?b - game_object)
        (and
          (forall (?p - hexagonal_bin ?x - block ?v - dodgeball)
            (and
              (exists (?m - dodgeball)
                (game-conserved
                  (object_orientation desk agent)
                )
              )
              (and
                (forall (?u - hexagonal_bin)
                  (forall (?w - (either triangular_ramp yellow_cube_block))
                    (not
                      (and
                        (game-conserved
                          (on ?u)
                        )
                        (and
                          (game-conserved
                            (touch ?v)
                          )
                          (forall (?m - hexagonal_bin)
                            (game-conserved
                              (agent_holds ?w)
                            )
                          )
                          (exists (?y - dodgeball)
                            (or
                              (game-conserved
                                (not
                                  (touch ?w)
                                )
                              )
                              (game-optional
                                (and
                                  (and
                                    (in_motion ?u ?w)
                                    (not
                                      (agent_holds ?w)
                                    )
                                  )
                                  (object_orientation bridge_block ?v)
                                  (and
                                    (forall (?m - block)
                                      (on ?u ?w)
                                    )
                                    (not
                                      (in ?v rug)
                                    )
                                  )
                                )
                              )
                              (game-optional
                                (agent_holds ?v ?y)
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (exists (?u - rug ?t - ball)
                  (forall (?j - hexagonal_bin ?i ?f ?s - dodgeball)
                    (exists (?q - game_object ?c - cube_block ?r - golfball ?e - hexagonal_bin)
                      (and
                        (and
                          (game-conserved
                            (on ?f)
                          )
                        )
                        (game-conserved
                          (in upright ?v)
                        )
                        (and
                          (forall (?d - ball)
                            (game-optional
                              (not
                                (agent_holds desk ?v)
                              )
                            )
                          )
                          (and
                            (forall (?r - game_object ?n - hexagonal_bin)
                              (forall (?g ?o - (either alarm_clock pyramid_block))
                                (forall (?x - ball ?k - ball ?w - (either chair yellow_cube_block dodgeball))
                                  (and
                                    (and
                                      (forall (?d - hexagonal_bin)
                                        (and
                                          (game-optional
                                            (and
                                              (and
                                                (not
                                                  (agent_holds ?o)
                                                )
                                                (and
                                                  (touch ?w ?i)
                                                  (touch ?o)
                                                  (in_motion bed ?s)
                                                )
                                              )
                                              (< (distance agent desk) 1)
                                            )
                                          )
                                          (and
                                            (forall (?c - ball ?z - triangular_ramp ?p - triangular_ramp)
                                              (exists (?u - dodgeball)
                                                (game-conserved
                                                  (touch ?s ?d)
                                                )
                                              )
                                            )
                                          )
                                          (game-conserved
                                            (agent_holds ?d ?t)
                                          )
                                        )
                                      )
                                    )
                                  )
                                )
                              )
                            )
                            (and
                              (forall (?l - hexagonal_bin)
                                (forall (?g - building)
                                  (forall (?z - color ?o ?j ?h - game_object)
                                    (or
                                      (game-conserved
                                        (agent_holds ?b)
                                      )
                                      (and
                                        (exists (?d - doggie_bed)
                                          (and
                                            (and
                                              (game-optional
                                                (in_motion ?b agent)
                                              )
                                              (or
                                                (and
                                                  (game-conserved
                                                    (in ?o)
                                                  )
                                                  (exists (?w - cube_block)
                                                    (exists (?k - (either golfball wall))
                                                      (game-conserved
                                                        (in_motion agent)
                                                      )
                                                    )
                                                  )
                                                  (and
                                                    (and
                                                      (exists (?p - blinds ?m ?a - yellow_cube_block)
                                                        (forall (?w - desktop ?r - wall)
                                                          (or
                                                            (game-conserved
                                                              (and
                                                                (on ?r door)
                                                                (touch ?m agent)
                                                              )
                                                            )
                                                            (forall (?c - wall)
                                                              (game-conserved
                                                                (in_motion ?h ?c)
                                                              )
                                                            )
                                                            (game-conserved
                                                              (in ?r)
                                                            )
                                                          )
                                                        )
                                                      )
                                                    )
                                                    (and
                                                      (game-optional
                                                        (on ?g ?e)
                                                      )
                                                    )
                                                    (exists (?u - dodgeball)
                                                      (game-conserved
                                                        (on ?o)
                                                      )
                                                    )
                                                    (game-optional
                                                      (agent_holds ?e ?g)
                                                    )
                                                    (game-conserved
                                                      (not
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
                                    )
                                  )
                                )
                              )
                              (game-conserved
                                (agent_holds ?f)
                              )
                              (and
                                (and
                                  (and
                                    (game-optional
                                      (in_motion ?i ?f)
                                    )
                                    (game-optional
                                      (agent_holds ?f)
                                    )
                                    (game-optional
                                      (not
                                        (and
                                          (in ?i ?v)
                                          (and
                                            (in_motion ?v)
                                            (agent_holds ?v)
                                          )
                                        )
                                      )
                                    )
                                  )
                                )
                                (game-conserved
                                  (touch ?b)
                                )
                              )
                            )
                          )
                          (forall (?h ?q ?d ?p - red_dodgeball ?w - ball)
                            (and
                              (game-conserved
                                (agent_holds ?w)
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (forall (?w - ball ?k - flat_block)
                  (game-optional
                    (in_motion ?v ?k)
                  )
                )
              )
              (and
                (exists (?o - hexagonal_bin)
                  (game-optional
                    (agent_holds ?b ?v)
                  )
                )
                (game-conserved
                  (and
                    (not
                      (in ?v ?v)
                    )
                    (touch ?b agent)
                  )
                )
              )
            )
          )
        )
      )
      (forall (?m - hexagonal_bin ?s - teddy_bear)
        (exists (?d - doggie_bed)
          (exists (?r ?j ?v - hexagonal_bin)
            (exists (?u - chair)
              (exists (?t - pillow)
                (and
                  (game-conserved
                    (agent_holds ?s ?v)
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
    (forall (?g - wall)
      (and
        (preference preference1
          (exists (?b - dodgeball ?u - building)
            (exists (?c - teddy_bear)
              (exists (?v - dodgeball)
                (at-end
                  (and
                    (and
                      (adjacent bed)
                      (exists (?i - ball)
                        (in_motion ?v)
                      )
                    )
                    (and
                      (< (distance room_center ?v) 1)
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
      (exists (?h - dodgeball ?f - dodgeball)
        (then
          (once (agent_holds ?f) )
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
  (exists (?b - hexagonal_bin ?w ?f - dodgeball)
    (exists (?a - ball)
      (game-conserved
        (adjacent ?w)
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - block ?u - hexagonal_bin ?p - teddy_bear ?l - (either laptop pink chair dodgeball bridge_block pyramid_block desktop))
      (and
        (preference preference1
          (then
            (hold (in_motion ?l ?l ?l) )
            (hold (and (in_motion pillow) (on ?l) ) )
            (once (and (agent_holds ?l) (in agent) (on ?l) ) )
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
    (forall (?p - flat_block)
      (exists (?k - teddy_bear ?k - sliding_door)
        (exists (?m - doggie_bed)
          (exists (?n - color ?o ?n - building)
            (game-optional
              (and
                (not
                  (not
                    (in_motion ?o)
                  )
                )
                (<= (distance ?m ?p) (distance ?o ?k))
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
      (exists (?s - hexagonal_bin)
        (exists (?u - yellow_cube_block ?b - hexagonal_bin)
          (then
            (once (in_motion front) )
            (once (and (in_motion ?b) ) )
            (hold (not (exists (?y - block ?q - doggie_bed ?c - hexagonal_bin) (in_motion front ?s) ) ) )
          )
        )
      )
    )
    (forall (?e - ball ?s - (either watch cube_block hexagonal_bin))
      (and
        (preference preference2
          (exists (?z - (either chair side_table) ?x - curved_wooden_ramp)
            (then
              (hold (exists (?c ?w - dodgeball ?u - doggie_bed) (in ?s ?x) ) )
              (once-measure (object_orientation ) (distance ?s ?x) )
              (once (not (not (in_motion ?x) ) ) )
              (once (adjacent desk ?x) )
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
    (forall (?w ?a ?t ?o - chair)
      (forall (?c - dodgeball)
        (and
          (exists (?i - (either bridge_block dodgeball triangle_block yellow_cube_block))
            (game-optional
              (and
                (on ?t)
                (not
                  (in ?w ?a)
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
      (exists (?c - ball)
        (exists (?t - curved_wooden_ramp ?g - hexagonal_bin)
          (exists (?k ?b - hexagonal_bin)
            (exists (?v - ball)
              (exists (?a - (either cylindrical_block dodgeball) ?x - cube_block)
                (at-end
                  (and
                    (not
                      (not
                        (not
                          (in ?c ?g)
                        )
                      )
                    )
                    (not
                      (and
                        (in_motion bed)
                        (not
                          (exists (?j - dodgeball)
                            (not
                              (not
                                (and
                                  (not
                                    (agent_holds ?j)
                                  )
                                  (in ?x)
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
      (exists (?d ?x ?i - hexagonal_bin)
        (then
          (hold (in_motion ?d) )
          (hold-while (agent_holds desk) (in_motion ?i) )
          (hold (in ?d) )
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
      (exists (?o - building)
        (forall (?v - building ?i - game_object ?l - (either beachball pyramid_block dodgeball pink laptop tall_cylindrical_block lamp))
          (exists (?i - triangular_ramp)
            (at-end
              (and
                (same_object ?i ?l)
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
  (exists (?o - dodgeball)
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
  (forall (?h - curved_wooden_ramp)
    (game-conserved
      (between ?h)
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
        (once (exists (?e - (either bridge_block golfball) ?a - dodgeball) (in ?a ?a) ) )
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
    (forall (?o ?v ?g ?e ?n ?q - shelf)
      (and
        (preference preference3
          (then
            (once (on ?v ?e) )
            (once (not (in ?n) ) )
            (hold (in agent ?e) )
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
    (forall (?h ?d ?m - block)
      (and
        (preference preference1
          (then
            (once (not (in_motion ?m) ) )
            (hold-while (and (and (adjacent ?d) (not (and (in_motion ?d ?d) (on ?h) (agent_holds ?h) ) ) (and (game_start ?d) (touch ?d) (or (in ?h) (not (not (not (touch floor) ) ) ) ) ) (agent_holds agent) ) (agent_holds ?h) (agent_holds agent) ) (same_color ?m) )
            (hold-while (on pink_dodgeball ?h) (< 1 10) (on pink_dodgeball ?d) )
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
      (exists (?e - (either cd blue_cube_block alarm_clock))
        (then
          (once (in_motion ?e) )
          (once (object_orientation ?e) )
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
  (forall (?b ?a - dodgeball)
    (and
      (and
        (and
          (game-conserved
            (not
              (in ?b)
            )
          )
          (game-conserved
            (not
              (not
                (adjacent ?b ?a)
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
    (forall (?r ?q ?y ?b - book ?k - block)
      (and
        (preference preference1
          (then
            (hold (and (not (rug_color_under ?k) ) (forall (?v - (either pen doggie_bed)) (adjacent ?k ?v) ) ) )
            (once (in ?k) )
            (hold (agent_holds ?k ?k) )
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
  (forall (?e - shelf)
    (game-conserved
      (between ?e ?e)
    )
  )
)
(:constraints
  (and
    (forall (?v - hexagonal_bin ?h - book)
      (and
        (preference preference1
          (then
            (once (and (on ?h ?h) (and (not (in_motion south_wall) ) (in_motion agent ?h) ) ) )
            (once (forall (?l - drawer) (exists (?m - pyramid_block) (agent_holds ?l) ) ) )
            (hold-while (in_motion ?h agent) (not (not (agent_holds ?h ?h) ) ) )
          )
        )
        (preference preference2
          (exists (?g - hexagonal_bin)
            (exists (?b - wall ?k - doggie_bed)
              (at-end
                (agent_holds ?g)
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
    (exists (?c - teddy_bear ?x - watch)
      (exists (?r - (either game_object game_object alarm_clock) ?w - triangular_ramp ?v - game_object ?c - hexagonal_bin)
        (game-conserved
          (exists (?p - cube_block ?b - cube_block)
            (and
              (and
                (exists (?u - cube_block)
                  (in_motion ?b)
                )
                (in_motion ?c ?c)
              )
              (agent_holds ?c ?x)
              (touch ?x)
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
        (once (and (forall (?v - pillow ?u - building) (in_motion ?u) ) (in_motion ?xxx ?xxx) ) )
      )
    )
    (preference preference2
      (exists (?p - ball)
        (at-end
          (adjacent ?p)
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
  (exists (?y - color)
    (and
      (exists (?p - hexagonal_bin)
        (game-optional
          (and
            (and
              (in_motion agent)
              (not
                (touch front bed)
              )
              (in ?p ?y)
            )
            (in ?y)
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
              (exists (?r - building)
                (agent_holds ?r front)
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
  (exists (?p - hexagonal_bin)
    (and
      (forall (?n - ball)
        (and
          (and
            (and
              (forall (?s - block)
                (or
                  (game-optional
                    (and
                      (agent_holds ?s)
                      (exists (?z - dodgeball)
                        (not
                          (on agent ?s)
                        )
                      )
                      (adjacent ?p)
                    )
                  )
                )
              )
              (and
                (game-conserved
                  (and
                    (on ?p ?p)
                    (not
                      (agent_holds ?n ?p)
                    )
                    (not
                      (not
                        (not
                          (game_over ?n ?p)
                        )
                      )
                    )
                  )
                )
                (and
                  (game-conserved
                    (agent_holds ?n)
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
            (in_motion ?p)
          )
          (on ?p)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - hexagonal_bin)
        (at-end
          (same_type ?u ?u)
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
    (exists (?d - hexagonal_bin ?o - chair)
      (game-optional
        (in_motion ?o)
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
        (once (not (and (and (in_motion yellow) (not (< (distance 4 ?xxx ?xxx) 6) ) ) (agent_holds ?xxx ?xxx) (and (in ?xxx ?xxx) (forall (?r - hexagonal_bin) (on ?r) ) ) ) ) )
      )
    )
    (preference preference2
      (exists (?v - block ?v - teddy_bear)
        (then
          (once (touch ?v) )
          (once (object_orientation ?v ?v) )
          (hold (in_motion rug) )
        )
      )
    )
    (forall (?j - shelf)
      (and
        (preference preference3
          (forall (?h ?b - golfball)
            (exists (?k - cube_block ?a - dodgeball)
              (then
                (once (in_motion agent side_table) )
                (once (= 1 7) )
                (once (not (and (agent_holds ?j ?j) (and (not (adjacent ?h) ) (not (rug_color_under ?j ?h) ) (on ?h) ) ) ) )
              )
            )
          )
        )
        (preference preference4
          (then
            (once (not (not (agent_holds agent) ) ) )
            (hold (agent_holds ?j top_shelf) )
            (hold (agent_holds ?j ?j) )
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
      (exists (?d ?r - (either cellphone cube_block) ?f - blue_pyramid_block)
        (exists (?b - pyramid_block)
          (then
            (once (not (agent_holds ?f ?f) ) )
            (hold-while (not (exists (?l - dodgeball) (not (on bed ?l) ) ) ) (and (forall (?n - flat_block ?n - dodgeball) (on ?f ?n) ) (adjacent ?f) ) )
            (once (on rug ?f) )
          )
        )
      )
    )
    (preference preference2
      (exists (?f - (either desktop dodgeball) ?i - book)
        (then
          (hold (in_motion ?i ?i ?i) )
          (hold (not (agent_holds ?i) ) )
          (once (not (not (or (in_motion ?i) (agent_holds ?i ?i) ) ) ) )
        )
      )
    )
    (preference preference3
      (exists (?c - triangular_ramp)
        (exists (?a - hexagonal_bin)
          (exists (?x - (either cellphone pen beachball dodgeball dodgeball cube_block chair))
            (then
              (once (< (distance ?a side_table agent) (distance ?a 2)) )
              (once (< 1 2) )
              (hold (exists (?q - hexagonal_bin) (not (and (not (and (and (not (not (on ?x ?x) ) ) (not (and (agent_holds ?a) (forall (?b - game_object) (not (agent_holds agent ?a) ) ) ) ) ) (adjacent_side ?a) ) ) (in ?a floor ?c) ) ) ) )
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
  (exists (?p - curved_wooden_ramp ?r - game_object ?d - block)
    (and
      (and
        (game-conserved
          (on ?d ?d)
        )
      )
      (game-optional
        (is_setup_object bed)
      )
      (forall (?n - color)
        (exists (?r - game_object)
          (exists (?c - cylindrical_block)
            (game-optional
              (exists (?e - ball)
                (not
                  (in_motion ?e ?r)
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
    (forall (?i - cylindrical_block)
      (and
        (preference preference1
          (then
            (once (agent_holds agent) )
            (once-measure (agent_holds ?i) (distance ?i room_center ?i) )
            (hold (same_object top_shelf ?i) )
            (hold-while (agent_holds ?i) (agent_holds ?i) )
          )
        )
      )
    )
    (forall (?e - block ?p - teddy_bear)
      (and
        (preference preference2
          (then
            (hold (in ?p) )
            (hold (not (and (on yellow) (agent_holds ?p) ) ) )
          )
        )
        (preference preference3
          (then
            (once (and (and (and (not (on ?p ?p) ) (on ?p desk) ) (agent_holds ?p ?p) ) (on ?p ?p) (not (in ?p) ) ) )
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
        (exists (?a - dodgeball ?t - pyramid_block)
          (or
            (and
              (game-conserved
                (and
                  (< (distance ?t ?t) (distance back 9))
                  (in_motion ?t)
                )
              )
            )
            (and
              (game-conserved
                (in_motion bridge_block)
              )
              (and
                (game-optional
                  (agent_holds ?t)
                )
                (exists (?m - hexagonal_bin)
                  (and
                    (exists (?j - (either pencil golfball yellow_cube_block))
                      (forall (?n - hexagonal_bin)
                        (exists (?l - ball)
                          (game-conserved
                            (same_type rug ?j)
                          )
                        )
                      )
                    )
                    (and
                      (forall (?c - block)
                        (forall (?r - dodgeball)
                          (and
                            (game-optional
                              (agent_holds ?r bed)
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (game-optional
                  (on ?t)
                )
              )
              (game-conserved
                (adjacent ?t)
              )
            )
          )
        )
        (forall (?b - shelf ?z - chair)
          (and
            (game-conserved
              (on ?z bed)
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
                (exists (?l - block)
                  (touch agent ?l)
                )
                (on ?xxx ?xxx)
              )
              (agent_holds ?xxx)
            )
            (not
              (or
                (and
                  (exists (?i - curved_wooden_ramp)
                    (not
                      (and
                        (not
                          (agent_holds ?i)
                        )
                        (not
                          (exists (?g - dodgeball ?x - cube_block ?r - dodgeball)
                            (on bed)
                          )
                        )
                        (agent_holds floor)
                      )
                    )
                  )
                  (exists (?e - hexagonal_bin)
                    (in ?e)
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
    (exists (?j - beachball ?w - block)
      (and
        (not
          (forall (?l - hexagonal_bin ?u - cube_block)
            (exists (?h - block)
              (and
                (exists (?x - hexagonal_bin ?f - (either bridge_block wall mug alarm_clock main_light_switch dodgeball game_object) ?l - ball)
                  (game-optional
                    (agent_holds ?u ?h)
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
      (exists (?a - wall)
        (then
          (once (and (touch ?a ?a) (toggled_on ?a ?a) (= (distance 1 ?a) 1) ) )
          (hold (toggled_on ?a) )
          (once (not (agent_holds ?a ?a) ) )
        )
      )
    )
    (preference preference2
      (exists (?v - hexagonal_bin ?u - cylindrical_block)
        (exists (?y - game_object)
          (exists (?s - wall)
            (at-end
              (agent_holds ?s agent)
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
  (forall (?g - (either doggie_bed wall))
    (game-conserved
      (not
        (in_motion ?g agent)
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - game_object)
      (and
        (preference preference1
          (exists (?u - dodgeball ?x - dodgeball ?d - (either dodgeball laptop))
            (exists (?v - hexagonal_bin)
              (exists (?z - bridge_block)
                (then
                  (hold-while (in_motion agent ?z) (touch ?z ?i) )
                  (hold (< (distance 0 ?d 4) 1) )
                  (hold (agent_holds ?d pink_dodgeball) )
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?y - curved_wooden_ramp)
        (exists (?q - chair)
          (then
            (hold-to-end (on ?q) )
            (hold (and (agent_holds ?q ?y) (opposite ?q ?y) ) )
            (once (in ?y ?q) )
          )
        )
      )
    )
    (preference preference3
      (exists (?s - dodgeball ?p - dodgeball ?c - hexagonal_bin)
        (exists (?z - chair ?l - (either curved_wooden_ramp golfball))
          (then
            (once (not (touch agent ?l) ) )
            (once (< 1 (distance 5 ?c)) )
            (once (not (and (in_motion ?c) (in_motion ?l ?l) ) ) )
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
    (exists (?i - hexagonal_bin ?i - hexagonal_bin)
      (exists (?h - building ?o - block ?k - hexagonal_bin ?e - hexagonal_bin ?y - hexagonal_bin)
        (game-optional
          (not
            (adjacent ?y)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?v - dodgeball)
      (and
        (preference preference1
          (then
            (hold (not (in ?v) ) )
            (once (is_setup_object rug) )
            (hold (in_motion ?v ?v) )
          )
        )
      )
    )
    (preference preference2
      (exists (?r - building)
        (exists (?g - wall ?y - block)
          (exists (?f - (either blue_cube_block pillow) ?e - dodgeball)
            (exists (?g - book ?i - color)
              (then
                (once-measure (on ?y) (distance door green_golfball) )
                (once (in_motion ?e ?y) )
                (once (in_motion ?e ?e) )
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
      (exists (?s - curved_wooden_ramp)
        (then
          (once (not (< (distance ?s ?s) (distance ?s ?s)) ) )
          (once (rug_color_under ?s) )
          (hold-while (agent_holds ?s agent) (and (not (not (adjacent ?s ?s) ) ) (not (and (> 2 (distance ?s back)) (object_orientation ?s ?s) ) ) ) (< 0.5 (x_position 8 8)) (agent_holds pink_dodgeball) )
        )
      )
    )
    (forall (?s ?g ?t ?a - game_object)
      (and
        (preference preference2
          (exists (?u - game_object ?z - block ?i - dodgeball ?p - wall)
            (then
              (once (agent_holds ?s) )
              (hold (on ?s) )
              (once (in_motion ?s) )
            )
          )
        )
        (preference preference3
          (then
            (hold (not (not (not (agent_holds ?s) ) ) ) )
            (once (not (exists (?u - cube_block ?l - tall_cylindrical_block) (not (in ?l ?s) ) ) ) )
            (once (on ?a) )
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
      (exists (?x - curved_wooden_ramp)
        (in_motion ?x)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (and (not (rug_color_under ?xxx ?xxx) ) (< 2 (distance ?xxx desk)) (and (agent_holds ?xxx ?xxx) (not (and (adjacent ?xxx) (adjacent_side ?xxx front) (exists (?o - hexagonal_bin ?i - doggie_bed ?e - drawer) (in_motion ?e left) ) (in ?xxx ?xxx) (and (agent_holds ?xxx) (touch ?xxx ?xxx) ) (agent_holds ?xxx) ) ) ) ) (forall (?b - (either cylindrical_block pyramid_block golfball) ?x - building) (not (and (and (agent_holds ?x) (and (in_motion ?x ?x) (not (and (< (x_position 3 ?x) (distance ?x)) (adjacent ?x) ) ) ) ) (and (touch floor) (in ?x ?x) ) ) ) ) ) )
        (once (exists (?u - ball) (and (in_motion desk) (< 10 1) ) ) )
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
    (forall (?z - cube_block)
      (and
        (preference preference2
          (then
            (once (in ?z) )
            (once (same_type ?z rug) )
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
    (forall (?w - block)
      (and
        (preference preference1
          (then
            (hold (and (agent_holds ?w) (on ?w) (and (on ?w desk) (agent_holds ?w ?w) ) (agent_holds pink) ) )
            (once (and (agent_holds ?w) (and (open floor) (and (touch ?w) (touch ?w ?w) ) ) ) )
            (once (not (not (in_motion ?w) ) ) )
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
  (forall (?s - game_object)
    (not
      (forall (?t - dodgeball ?y - triangular_ramp ?j - hexagonal_bin)
        (game-optional
          (not
            (in_motion ?s ?j)
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
      (exists (?j - triangular_ramp ?e - pillow)
        (then
          (once (agent_holds ?e ?e) )
          (hold (agent_holds ?e ?e desk) )
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
  (forall (?b - cube_block)
    (exists (?h ?a - curved_wooden_ramp)
      (exists (?q - teddy_bear)
        (and
          (exists (?j - chair)
            (game-optional
              (toggled_on ?j)
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
              (exists (?m - building ?m - desk_shelf ?h - wall ?o - chair)
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
                              (exists (?b - (either golfball lamp))
                                (adjacent ?b)
                              )
                              (and
                                (>= (distance_side room_center ?xxx) (distance ?xxx ?xxx))
                                (exists (?m - dodgeball)
                                  (and
                                    (in ?m ?m)
                                    (in_motion ?m pink_dodgeball)
                                    (and
                                      (on ?m)
                                      (and
                                        (or
                                          (and
                                            (not
                                              (in_motion ?m ?m)
                                            )
                                            (in_motion ?m)
                                          )
                                          (not
                                            (agent_holds ?m)
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
                          (exists (?t - hexagonal_bin ?v - cylindrical_block ?r - cube_block)
                            (touch ?r)
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
          (exists (?t - (either dodgeball pyramid_block))
            (game-conserved
              (in_motion ?t)
            )
          )
        )
      )
    )
    (not
      (exists (?v - ball)
        (game-conserved
          (forall (?y - drawer ?h - doggie_bed)
            (not
              (agent_holds ?h ?h)
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
      (exists (?g - hexagonal_bin)
        (exists (?i - hexagonal_bin ?m - curved_wooden_ramp)
          (then
            (once (agent_holds ?m) )
            (hold (agent_holds desk) )
            (once (and (and (not (and (not (agent_holds ?g) ) (in_motion ?m) ) ) ) ) )
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
                (exists (?o - shelf)
                  (not
                    (in_motion ?o)
                  )
                )
              )
              (between ?xxx)
            )
          )
        )
      )
    )
    (forall (?k - (either cylindrical_block rug) ?a ?g - ball)
      (and
        (preference preference4
          (then
            (hold (in_motion ?g ?a) )
            (any)
            (once (in_motion ?g ?g) )
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
      (exists (?o - shelf ?h - hexagonal_bin)
        (exists (?k - hexagonal_bin)
          (at-end
            (touch ?k)
          )
        )
      )
    )
    (preference preference2
      (exists (?g - yellow_pyramid_block ?l - (either book yellow_cube_block))
        (exists (?k ?j - dodgeball)
          (then
            (once (< 9 (distance door bed)) )
            (once (= (distance ?l ?l) (distance ?k ?j)) )
            (once (adjacent ?j) )
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
      (exists (?a - doggie_bed)
        (exists (?t - (either key_chain desktop key_chain yellow_cube_block cd yellow doggie_bed))
          (exists (?f - (either golfball tall_cylindrical_block) ?u - teddy_bear)
            (then
              (once (not (not (agent_holds ?u) ) ) )
              (once (not (not (and (on ?t ?u) (adjacent ?u ?t) ) ) ) )
              (hold (agent_holds ?u ?t) )
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
      (exists (?t - (either tall_cylindrical_block golfball) ?x - tall_cylindrical_block)
        (and
          (and
            (game-optional
              (not
                (agent_holds ?x)
              )
            )
          )
          (forall (?h - hexagonal_bin)
            (game-conserved
              (not
                (not
                  (in_motion ?h agent)
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
      (exists (?u ?d - block ?e ?c - drawer)
        (exists (?n - game_object)
          (then
            (hold (agent_holds agent) )
            (hold (not (not (exists (?b - hexagonal_bin) (in ?n) ) ) ) )
            (once (adjacent ?e) )
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
    (forall (?k - shelf)
      (and
        (preference preference2
          (then
            (hold-while (not (adjacent ?k) ) (in_motion ?k ?k) )
            (once (adjacent ?k ?k) )
            (once (agent_holds ?k ?k) )
          )
        )
        (preference preference3
          (then
            (once (agent_holds agent) )
            (hold-while (and (not (and (not (and (in_motion ?k) (not (and (in_motion ?k ?k) (in agent) ) ) ) ) (in ?k) ) ) (not (agent_holds ?k green) ) ) (agent_holds ?k ?k) )
            (once (not (between ?k ?k) ) )
          )
        )
      )
    )
    (forall (?w - hexagonal_bin)
      (and
        (preference preference4
          (exists (?l - wall)
            (exists (?c - (either doggie_bed curved_wooden_ramp) ?k - dodgeball ?j - pillow)
              (then
                (once (< (distance ?w ?l) 0) )
                (hold (not (and (or (agent_holds ?w ?w) (not (in_motion ?l ?j) ) ) (same_type desk ?w) (exists (?x - hexagonal_bin) (and (adjacent ?x) (and (not (forall (?h ?f - hexagonal_bin) (agent_holds ?h) ) ) (not (agent_holds ?x ?w) ) ) (not (in_motion agent) ) ) ) ) ) )
                (hold (rug_color_under ?j) )
              )
            )
          )
        )
        (preference preference5
          (exists (?e - doggie_bed ?m - doggie_bed)
            (exists (?d - block ?p - block)
              (then
                (any)
                (hold-while (and (in_motion ?w ?p) (< 1 (distance 6 ?p)) ) (and (agent_holds ?w) (in ?p ?m) ) )
                (once (above ?m) )
                (once (in_motion ?p) )
                (once (adjacent ?m south_wall) )
                (once (not (exists (?j - dodgeball ?f - building) (agent_holds ?f) ) ) )
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
      (exists (?a - hexagonal_bin)
        (exists (?m - ball)
          (exists (?u - dodgeball)
            (exists (?q - (either cd cellphone) ?n - dodgeball ?j - building)
              (exists (?c ?d - teddy_bear)
                (exists (?p - hexagonal_bin ?z - hexagonal_bin ?w ?l ?f - dodgeball)
                  (exists (?k - dodgeball ?s - hexagonal_bin ?e - hexagonal_bin)
                    (then
                      (once (on ?f) )
                      (once (exists (?p - shelf) (< 1 (distance ?p)) ) )
                      (once (agent_holds ?u ?d) )
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
  (forall (?n - hexagonal_bin)
    (exists (?p - (either cube_block pyramid_block) ?o - red_pyramid_block)
      (and
        (and
          (and
            (game-conserved
              (in ?o ?n ?o)
            )
          )
          (game-conserved
            (in_motion ?n bed)
          )
          (not
            (game-conserved
              (adjacent ?o)
            )
          )
        )
        (game-optional
          (agent_holds floor ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?z - hexagonal_bin)
      (and
        (preference preference1
          (exists (?x - game_object)
            (exists (?q - shelf ?v - hexagonal_bin)
              (at-end
                (and
                  (agent_holds ?v)
                  (or
                    (not
                      (in ?v)
                    )
                    (between ?x ?z)
                  )
                )
              )
            )
          )
        )
        (preference preference2
          (then
            (once (in ?z) )
            (once (agent_holds ?z ?z) )
            (hold (and (on ?z tan) (and (and (agent_holds ?z) (in_motion ?z ?z) ) (and (not (agent_holds ?z) ) (in_motion ?z ?z) ) ) (in ?z agent) (and (< (distance 6 9) (distance ?z 10)) (not (in_motion ?z ?z) ) ) ) )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (agent_holds ?xxx ?xxx)
      )
    )
    (forall (?e - dodgeball ?i - (either golfball yellow book))
      (and
        (preference preference4
          (then
            (once (in ?i ?i) )
            (once (not (or (and (and (opposite floor) (in_motion ?i) (not (not (and (adjacent_side ?i ?i) (not (touch ?i) ) ) ) ) ) (and (and (agent_holds ?i ?i) (not (in ?i ?i ?i) ) ) (agent_holds ?i) (in ?i ?i) ) ) (in_motion ?i) ) ) )
            (once (object_orientation ?i ?i) )
          )
        )
        (preference preference5
          (at-end
            (agent_holds ?i)
          )
        )
        (preference preference6
          (exists (?t - pillow)
            (then
              (once (and (not (and (not (< (building_size 8 ?t) 1) ) (and (exists (?c - (either basketball dodgeball dodgeball)) (agent_holds ?c) ) (not (or (and (not (not (in_motion bed) ) ) (in floor) ) (on ?i ?t) (in_motion rug) ) ) (not (not (in_motion ?i ?i) ) ) ) ) ) (in ?i ?i) ) )
              (once (not (and (in_motion ?t upright) (not (and (agent_holds ?i) (on pink_dodgeball) (not (exists (?e - (either dodgeball alarm_clock) ?e - color) (in_motion ?t) ) ) (and (not (= (distance agent ?i) 4) ) (and (agent_holds ?t ?t) (on ?t) (agent_holds ?i door) (adjacent ?i) (not (in ) ) (exists (?f - cube_block) (not (in_motion agent ?i) ) ) ) ) ) ) ) ) )
              (once (in_motion ?t ?i) )
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
        (hold-while (forall (?i - chair) (agent_holds ?i floor) ) (in ?xxx ?xxx) )
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
  (exists (?n - (either yellow_cube_block cube_block cube_block) ?z - block)
    (and
      (forall (?w - chair)
        (exists (?v - triangular_ramp)
          (game-conserved
            (on ?v ?z)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - ball)
        (exists (?g - hexagonal_bin)
          (exists (?d - block)
            (then
              (hold (not (<= 1 (distance room_center ?h)) ) )
              (once (and (agent_holds ?h ?g) (not (agent_holds ?d) ) ) )
              (hold (adjacent ?d ?d) )
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
        (once (exists (?v ?g - game_object) (in ?v ?g) ) )
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
        (hold-while (and (and (in_motion ?xxx desk) (agent_holds ?xxx) ) (and (not (in_motion ?xxx ?xxx) ) (and (touch ?xxx) (forall (?n - game_object) (agent_holds ?n) ) ) ) ) (and (in_motion ?xxx ball) (and (and (and (agent_holds ?xxx ?xxx) (not (not (or (< 1 (distance room_center 1)) (agent_holds ?xxx ?xxx) ) ) ) ) (agent_holds ?xxx ?xxx) (and (on ?xxx) (touch ?xxx floor) ) (on agent) ) (adjacent bed ?xxx) ) ) (exists (?i - doggie_bed) (on ?i) ) )
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
      (exists (?y - (either key_chain laptop) ?f - cube_block ?m - blinds ?o - dodgeball ?s - hexagonal_bin)
        (exists (?h - cube_block)
          (exists (?z - hexagonal_bin)
            (exists (?p - doggie_bed)
              (exists (?v - hexagonal_bin)
                (forall (?b - hexagonal_bin)
                  (exists (?j - hexagonal_bin)
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
      (exists (?m - dodgeball ?x - game_object)
        (at-end
          (agent_holds ?x)
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
      (exists (?s ?u ?e - wall)
        (at-end
          (not
            (not
              (in ?e ?e)
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
  (forall (?h - (either cd) ?r - book ?q - wall ?x - dodgeball)
    (and
      (and
        (and
          (and
            (exists (?l ?b ?c - cube_block ?t - hexagonal_bin)
              (forall (?i ?y - ball)
                (game-conserved
                  (and
                    (open ?x ?y)
                    (equal_z_position floor)
                  )
                )
              )
            )
          )
        )
        (game-conserved
          (in_motion ?x ?x)
        )
      )
      (or
        (exists (?s - block ?r - building ?r - hexagonal_bin ?y - teddy_bear)
          (forall (?o - doggie_bed ?o - dodgeball)
            (and
              (exists (?u - building)
                (forall (?s - hexagonal_bin ?w ?m - triangular_ramp)
                  (and
                    (game-conserved
                      (and
                        (not
                          (in_motion ?y)
                        )
                        (on ?w)
                      )
                    )
                    (forall (?b - doggie_bed)
                      (game-conserved
                        (and
                          (agent_holds bed)
                          (faces ?w green_golfball)
                          (not
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
        (game-conserved
          (in_motion ?x ?x)
        )
        (game-conserved
          (object_orientation ?x)
        )
      )
      (and
        (and
          (forall (?c - building)
            (exists (?l - cube_block ?j - hexagonal_bin ?e - (either teddy_bear mug))
              (game-optional
                (and
                  (not
                    (on ?x)
                  )
                  (and
                    (on ?x ?e)
                    (agent_holds ?e)
                    (not
                      (and
                        (agent_holds ?c)
                        (and
                          (and
                            (< (distance ?x 0) 1)
                            (in_motion rug ?x)
                            (and
                              (in ?x)
                              (<= 1 0.5)
                              (on ?x)
                            )
                          )
                          (adjacent ?e)
                        )
                      )
                    )
                    (forall (?f - hexagonal_bin)
                      (agent_holds agent ?f)
                    )
                    (not
                      (and
                        (in_motion ?c ?e)
                        (agent_holds desk)
                        (in_motion ?c ?c ?x)
                        (not
                          (in_motion ?x)
                        )
                        (agent_holds ?c)
                        (and
                          (not
                            (above agent ?e)
                          )
                          (in_motion upright)
                        )
                      )
                    )
                    (in_motion ?x ?c)
                    (not
                      (on ?x)
                    )
                    (adjacent ?e ?e)
                    (on agent)
                    (agent_holds ?x ?c)
                  )
                )
              )
            )
          )
          (exists (?a - teddy_bear)
            (not
              (exists (?j - dodgeball)
                (game-optional
                  (not
                    (and
                      (on ?j)
                      (forall (?g - hexagonal_bin)
                        (exists (?l - doggie_bed ?s - flat_block ?n - ball)
                          (object_orientation ?n)
                        )
                      )
                      (and
                        (forall (?p ?y - curved_wooden_ramp)
                          (< 2 3)
                        )
                        (not
                          (and
                            (or
                              (agent_holds ?a ?a)
                              (agent_holds ?x ?a)
                            )
                            (not
                              (agent_holds ?a)
                            )
                            (exists (?e - teddy_bear)
                              (not
                                (in ?x ?e)
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
              (opposite ?x ?x)
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
      (exists (?n - block ?f - block ?j - ball)
        (then
          (hold (or (agent_holds rug ?j) ) )
          (once (not (agent_holds floor ?j ?j) ) )
        )
      )
    )
    (preference preference2
      (then
        (hold (and (and (< 2 (distance room_center agent)) (not (not (in_motion ?xxx ?xxx) ) ) ) (in_motion ?xxx) (in ?xxx ?xxx) ) )
        (once (agent_holds ?xxx ?xxx) )
        (once-measure (in ?xxx ?xxx) (building_size room_center ?xxx) )
        (once (and (on ?xxx) (and (agent_holds ?xxx ?xxx) (exists (?m - curved_wooden_ramp) (and (and (and (and (and (in ?m ?m) (< 0.5 (distance ?m ?m)) ) (between ?m ?m) ) (between ?m ?m) (adjacent ?m ?m) (in_motion rug) ) (not (< 8 (distance 6 ?m)) ) (and (and (agent_holds bed) (or (adjacent_side ?m ?m) (forall (?a - (either bridge_block key_chain) ?b - hexagonal_bin) (adjacent ?b ?m) ) ) ) (and (object_orientation ?m) (not (between ?m) ) ) ) ) (and (agent_holds desk ?m) (not (and (and (or (agent_holds ?m ?m) ) (not (not (in_motion ?m desk) ) ) ) (not (agent_holds bed) ) ) ) ) ) ) ) ) )
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
  (exists (?r - hexagonal_bin)
    (exists (?u - hexagonal_bin ?q - (either cube_block top_drawer) ?n - game_object)
      (exists (?w - (either pyramid_block laptop golfball))
        (and
          (game-conserved
            (and
              (on ?n ?r)
              (agent_holds ?w bed)
            )
          )
          (game-conserved
            (and
              (not
                (in_motion agent)
              )
              (not
                (same_type ?r ?r)
              )
              (not
                (agent_holds ?r rug)
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
    (forall (?x - hexagonal_bin)
      (and
        (preference preference1
          (exists (?w - ball ?k - game_object)
            (then
              (once (agent_holds ?k agent) )
              (once (and (agent_holds ?k ?k) (in ?x) ) )
              (hold (agent_holds ?k) )
            )
          )
        )
        (preference preference2
          (then
            (hold (agent_holds ?x) )
            (once (adjacent upright ?x) )
            (hold-while (and (agent_holds agent) (adjacent ?x) (agent_holds ?x) ) (on ?x) )
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
      (exists (?x - game_object)
        (then
          (once (in_motion ?x ?x) )
          (hold-while (not (and (is_setup_object ?x ?x) (on ?x) ) ) (agent_holds ?x) )
          (hold (in_motion upright ?x) )
        )
      )
    )
    (preference preference6
      (then
        (once (in_motion ?xxx ?xxx) )
        (once (and (agent_holds agent south_west_corner) (not (< 6 (distance ?xxx room_center)) ) (in_motion ?xxx) ) )
        (once (and (and (agent_holds ?xxx ?xxx) (not (not (not (not (not (forall (?f - hexagonal_bin) (not (not (on ?f) ) ) ) ) ) ) ) ) ) (agent_holds ?xxx) ) )
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
    (forall (?m - hexagonal_bin ?n - curved_wooden_ramp ?k - red_dodgeball)
      (and
        (preference preference1
          (exists (?i - cube_block ?n - chair)
            (exists (?p - block)
              (exists (?b - dodgeball ?w - hexagonal_bin ?z - hexagonal_bin)
                (exists (?s - game_object ?g - hexagonal_bin ?t - hexagonal_bin ?t ?h ?s - cube_block)
                  (then
                    (once (or (on ?k rug) (and (not (in_motion ?t) ) (not (in_motion ?n) ) ) ) )
                    (once (< (distance ?z ?h) 1) )
                    (once (agent_holds ?s) )
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
    (forall (?u - chair)
      (and
        (preference preference1
          (then
            (once (in_motion ?u) )
            (hold (exists (?f - cube_block ?d - pyramid_block) (agent_holds ?d) ) )
            (once (not (not (agent_holds ?u) ) ) )
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
  (exists (?u - cube_block ?u ?w ?e - dodgeball)
    (and
      (and
        (game-conserved
          (in_motion ?w)
        )
        (forall (?y - drawer)
          (and
            (game-conserved
              (in ?w)
            )
            (game-conserved
              (in_motion agent ?u)
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
      (exists (?r - hexagonal_bin)
        (exists (?t - pillow)
          (then
            (hold-while (agent_holds ?r) (and (in ?r) ) )
            (once (and (< (distance ?r 10) 0) ) )
            (once (not (agent_holds ?r) ) )
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
  (exists (?d - hexagonal_bin)
    (exists (?l - beachball ?m - block)
      (and
        (forall (?w - hexagonal_bin)
          (or
            (not
              (forall (?i - bridge_block)
                (exists (?a - dodgeball)
                  (game-conserved
                    (not
                      (and
                        (in_motion ?a)
                        (not
                          (adjacent block ?i)
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
                  (exists (?y - hexagonal_bin)
                    (and
                      (in_motion agent)
                      (in_motion agent ?y)
                    )
                  )
                  (not
                    (agent_holds ?w)
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
    (exists (?s - ball)
      (exists (?b - chair)
        (game-conserved
          (in ?b)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?j - hexagonal_bin)
      (and
        (preference preference1
          (at-end
            (and
              (agent_holds south_west_corner)
              (and
                (agent_holds ?j ?j)
                (agent_holds front ?j)
              )
            )
          )
        )
        (preference preference2
          (then
            (hold (in_motion ?j) )
            (once (or (in_motion ?j ?j) ) )
            (once (in_motion agent ?j) )
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


(define (game game-id-100) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (on ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - game_object)
        (then
          (once (between ?u) )
          (once (agent_holds ?u) )
          (once (in_motion ?u ?u) )
        )
      )
    )
    (forall (?h - building)
      (and
        (preference preference2
          (then
            (once (exists (?g - color ?c - teddy_bear) (in_motion ?c) ) )
            (hold (on ?h ?h) )
          )
        )
      )
    )
    (preference preference3
      (then
        (any)
        (hold (and (and (in_motion ?xxx ?xxx) (and (and (on ?xxx ?xxx) (and (in_motion upside_down ?xxx) (and (equal_z_position ?xxx) (agent_holds ?xxx) ) ) ) (and (on ?xxx agent) (in_motion ?xxx) ) ) ) (and (in_motion ?xxx) (> 7 6) (and (< (distance 5 2) 1) (in ?xxx) (in ?xxx) ) (not (on ?xxx ?xxx) ) ) (and (adjacent ?xxx desk ?xxx) (not (agent_holds ?xxx) ) (and (is_setup_object bed) ) (in_motion ?xxx) (exists (?p - hexagonal_bin) (is_setup_object bed ?p) ) (not (not (and (adjacent agent desktop) (agent_holds agent) ) ) ) (on ?xxx) ) ) )
        (once (not (in_motion ?xxx ?xxx) ) )
      )
    )
    (preference preference4
      (then
        (once (not (in ?xxx ?xxx) ) )
        (once (rug_color_under desktop brown) )
      )
    )
    (preference preference5
      (exists (?p - doggie_bed)
        (exists (?s - dodgeball ?q - (either book key_chain))
          (then
            (hold (agent_holds ?p ?q) )
            (once (and (and (exists (?w - hexagonal_bin) (in ?w) ) (agent_holds ?q) (in_motion ?p) ) (and (in_motion ?p) (adjacent_side desk) (or (not (adjacent ?p) ) (touch ?p ?q) ) ) (in_motion upright ?p) ) )
            (once (agent_holds ?p) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-overlapping preference2:pyramid_block) (count preference3:beachball) )
    (>= 10 7 )
  )
)
(:scoring
  (* 6 8 )
)
)


(define (game game-id-101) (:domain few-objects-room-v1)
(:setup
  (forall (?j - game_object ?h - hexagonal_bin ?w - ball ?f - chair)
    (forall (?e - doggie_bed ?r - hexagonal_bin)
      (forall (?y - beachball ?p - cube_block ?j - drawer ?y - chair)
        (and
          (exists (?i - hexagonal_bin)
            (game-conserved
              (and
                (and
                  (adjacent ?r)
                  (in ?f)
                )
                (adjacent ?r ?i)
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
    (forall (?x ?e - doggie_bed)
      (and
        (preference preference1
          (exists (?g - (either key_chain pencil bed))
            (exists (?d - cube_block)
              (then
                (once-measure (adjacent ?g ?d) (distance 5 ?d) )
                (once (and (on desk) (<= (distance room_center ?x) (x_position ?e ?g)) ) )
                (hold (in_motion ?x) )
              )
            )
          )
        )
        (preference preference2
          (exists (?w - dodgeball ?h - curved_wooden_ramp)
            (exists (?w - game_object)
              (at-end
                (in_motion ?w)
              )
            )
          )
        )
        (preference preference3
          (at-end
            (forall (?k - red_dodgeball)
              (in ?e)
            )
          )
        )
      )
    )
    (preference preference4
      (then
        (once (agent_holds ?xxx desk) )
        (once (not (and (and (agent_holds ?xxx) (not (or (agent_holds ?xxx ?xxx) (<= 1 (distance room_center ?xxx)) ) ) ) (agent_holds ?xxx ?xxx) (exists (?z - hexagonal_bin) (on ?z) ) ) ) )
        (once (on ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count preference3:triangle_block:green) 300 )
)
(:scoring
  (count preference3:pyramid_block)
)
)


(define (game game-id-102) (:domain few-objects-room-v1)
(:setup
  (and
    (game-optional
      (and
        (in_motion ?xxx)
      )
    )
    (exists (?x - game_object)
      (game-conserved
        (not
          (adjacent ?x)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (in ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (> 4 (* 3 (count preference1:rug) (* 2 (* (count-measure preference1:dodgeball:pink) (+ (count-once-per-objects preference1:hexagonal_bin) 3 )
          (count preference1:golfball)
          (count-once-per-external-objects preference1:green)
          3
        )
      )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-103) (:domain many-objects-room-v1)
(:setup
  (exists (?l - curved_wooden_ramp ?b ?p - (either bridge_block tall_cylindrical_block credit_card))
    (and
      (game-conserved
        (not
          (agent_holds ?p)
        )
      )
      (and
        (forall (?r - doggie_bed ?n - curved_wooden_ramp)
          (and
            (and
              (game-conserved
                (not
                  (in_motion ?p)
                )
              )
              (game-conserved
                (in_motion ?n ?b)
              )
              (game-conserved
                (in_motion ?b)
              )
            )
            (game-conserved
              (in_motion ?p)
            )
            (forall (?t - shelf)
              (game-optional
                (touch ?n ?t)
              )
            )
            (game-optional
              (and
                (or
                  (in agent)
                  (agent_holds ?n ?b)
                  (not
                    (agent_holds ?b ?n)
                  )
                )
                (in upright)
              )
            )
            (game-conserved
              (not
                (exists (?g - bridge_block ?t - (either pencil))
                  (< 6 (distance 4 ?p))
                )
              )
            )
          )
        )
        (game-conserved
          (exists (?t - ball)
            (and
              (agent_holds floor ?t)
              (in_motion ?t)
            )
          )
        )
      )
      (game-conserved
        (and
          (on ?b)
          (and
            (in_motion front ?b)
            (not
              (forall (?w - (either mug yellow dodgeball))
                (and
                  (and
                    (in_motion ?w)
                    (exists (?k - doggie_bed ?m ?k ?x - curved_wooden_ramp ?i ?k - building)
                      (not
                        (and
                          (in_motion ?k)
                          (adjacent_side ?k ?w)
                        )
                      )
                    )
                  )
                  (opposite ?p ?w)
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
      (exists (?y - (either cube_block))
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
  (and
    (or
      (or
        (>= (< (>= (count preference1:doggie_bed:blue_dodgeball) (* (count-shortest preference1:beachball) (count-once-per-objects preference1:red:red) 5 (* (/ (total-score) (* (+ (+ (count-longest preference1:blue_pyramid_block) 3 )
                        2
                      )
                      (count preference1:blue_dodgeball)
                    )
                  )
                  (not
                    (+ 3 (count-once-per-objects preference1:beachball:book) )
                  )
                )
              )
            )
            (count preference1:yellow)
          )
          (* (count-once-per-objects preference1:doggie_bed) 4 (* 6 (+ (count preference1:beachball) (count preference1:basketball) (count preference1:tan) )
            )
            18
            (* 1 (count preference1:beachball) 2 )
            (count-same-positions preference1:pyramid_block:doggie_bed)
          )
        )
        (>= (count preference1:block) (* (count-once preference1:block) (count preference1:dodgeball) (count-once-per-objects preference1:doggie_bed) (count preference1:dodgeball) )
        )
        (>= (count preference1:yellow) (count-once-per-external-objects preference1:pink_dodgeball:yellow:dodgeball) )
      )
    )
    (>= (or (* 60 (* (count-once preference1:golfball) )
        )
        (* (* 5 5 )
          (count-longest preference1:golfball)
        )
        (count-once-per-objects preference1:beachball:blue_cube_block)
      )
      (* (not (< (and (count preference1:triangle_block) ) 30 )
        )
        (count preference1:yellow)
      )
    )
    (>= 3 (count preference1:pink) )
  )
)
(:scoring
  (count-once-per-objects preference1:dodgeball)
)
)


(define (game game-id-104) (:domain many-objects-room-v1)
(:setup
  (exists (?i - block)
    (game-conserved
      (adjacent ?i ?i)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l ?b ?k ?w - chair)
        (then
          (hold (touch ?k) )
          (once (same_object ?w ?l) )
          (any)
        )
      )
    )
  )
)
(:terminal
  (>= (count-shortest preference1:doggie_bed) 3 )
)
(:scoring
  (* (count-once-per-objects preference1:red) (- (* (count preference1:blue:dodgeball) (- (count preference1:basketball) )
      )
    )
  )
)
)


(define (game game-id-105) (:domain few-objects-room-v1)
(:setup
  (and
    (game-optional
      (not
        (>= 9 9)
      )
    )
  )
)
(:constraints
  (and
    (forall (?z - desktop ?v - (either cube_block cellphone))
      (and
        (preference preference1
          (exists (?d - golfball)
            (then
              (hold (in ?v ?d) )
              (once (not (and (not (not (on ?v) ) ) (in_motion ?d) ) ) )
              (once (not (agent_holds ?d desk) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count-total preference1:red) (+ (count-once-per-objects preference1:dodgeball:basketball) (external-forall-maximize (count-once preference1:beachball) ) )
  )
)
(:scoring
  (* (* (* (= (count-once-per-objects preference1:doggie_bed) )
        (+ 1 (< (count-once preference1:dodgeball) 8 )
          (* (count preference1:beachball) (total-score) )
        )
      )
      20
    )
    (external-forall-minimize
      (+ 180 )
    )
  )
)
)


(define (game game-id-106) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (touch ?xxx)
  )
)
(:constraints
  (and
    (forall (?b - triangular_ramp ?o - dodgeball)
      (and
        (preference preference1
          (then
            (once (agent_holds ?o ball) )
            (once (and (touch ?o ?o) (agent_holds ?o) ) )
            (once (not (exists (?a - (either yellow_cube_block dodgeball)) (exists (?e - game_object) (and (not (and (on ?a) (and (touch ?a pink_dodgeball) (agent_holds ?a) ) ) ) ) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:dodgeball) (* 3 (* (* (total-score) (* (- (count-once-per-objects preference1:dodgeball) )
            (count preference1:beachball)
          )
        )
        3
      )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-107) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (forall (?t - block)
          (not
            (not
              (and
                (in_motion agent)
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
        (once (and (forall (?l - hexagonal_bin) (adjacent_side agent desk) ) (not (and (touch ?xxx ?xxx) (in ?xxx) (in_motion ?xxx ?xxx) ) ) ) )
      )
    )
    (forall (?o - beachball ?y - dodgeball)
      (and
        (preference preference2
          (then
            (hold-while (not (on ?y ?y) ) (and (not (adjacent bed agent) ) (or (and (in_motion ?y) (touch ?y floor) ) (and (not (in_motion floor) ) ) ) ) )
            (hold (agent_holds ?y) )
            (hold-while (in ?y) (agent_holds north_wall) )
          )
        )
        (preference preference3
          (exists (?f - hexagonal_bin)
            (exists (?q - hexagonal_bin ?q - (either book pencil) ?k - dodgeball ?a - dodgeball)
              (then
                (once (not (agent_holds ?f) ) )
                (once (object_orientation ?f) )
                (once (agent_holds ?a ?f) )
              )
            )
          )
        )
        (preference preference4
          (exists (?h ?a ?g - game_object)
            (then
              (once (and (not (in_motion ?h) ) (<= (distance ?h ?y) 0) ) )
              (once (not (< (distance ?g 3 3) (distance ?g ?h)) ) )
              (once (not (same_color top_shelf back) ) )
            )
          )
        )
      )
    )
    (preference preference5
      (exists (?s - (either doggie_bed basketball))
        (exists (?c - chair ?a - dodgeball)
          (then
            (once (and (in_motion ?s) (not (and (agent_holds ?a ?s) (or (exists (?m - cube_block ?m - ball ?t - beachball) (not (not (agent_holds bed) ) ) ) (in_motion ?a) ) (not (>= (distance_side 7 ?s) 7) ) (>= 1 (distance 0 3)) ) ) ) )
            (once (and (on ?a) (in_motion ?s ?a) ) )
            (hold-to-end (< 1 2) )
          )
        )
      )
    )
    (forall (?v - yellow_pyramid_block)
      (and
        (preference preference6
          (then
            (once (< (distance 2) (building_size 3 ?v)) )
            (once (agent_holds floor front) )
            (hold-for 3 (and (not (and (and (and (not (< (distance desk ?v 1) (distance 2 3)) ) (or (not (and (in_motion ?v) (agent_holds ?v) (= (distance ?v ?v) 1) ) ) (and (on ?v) (in ?v ?v) ) ) ) (< 4 1) ) (between agent) ) ) ) )
          )
        )
        (preference preference7
          (exists (?g - dodgeball ?h - doggie_bed)
            (exists (?c ?s - wall ?b - (either beachball key_chain))
              (forall (?n - wall ?p - curved_wooden_ramp)
                (then
                  (once (and (on ?v) (in_motion bed bed) ) )
                  (once (agent_holds ?p) )
                  (once (and (not (equal_x_position ?b ?v) ) (agent_holds ?p) ) )
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
    (>= (count-once-per-objects preference1:bed) (+ (count-once-per-objects preference2:basketball:red) (* 6 (count preference6:dodgeball) )
      )
    )
    (>= (count-once preference2:basketball) (count preference6) )
    (<= 2 (count-once-per-objects preference2:pink) )
  )
)
(:scoring
  6
)
)


(define (game game-id-108) (:domain medium-objects-room-v1)
(:setup
  (exists (?t ?k - (either key_chain golfball) ?z - (either pyramid_block basketball laptop main_light_switch) ?n - (either golfball golfball))
    (exists (?b - shelf ?p - building)
      (game-conserved
        (not
          (touch ?p ?p)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (forall (?r - ball)
        (then
          (once (in_motion ?r) )
          (hold (on ?r) )
          (once (not (on ?r) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (- 10 )
  )
)
(:scoring
  (count preference1:green)
)
)


(define (game game-id-109) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - curved_wooden_ramp)
        (exists (?y - (either cellphone cellphone) ?z - shelf ?p - teddy_bear)
          (then
            (hold (and (and (and (in_motion ?s ?s) (in ?p ?s) ) (agent_holds ?p) ) (or (and (exists (?j - doggie_bed) (and (in ?p) (not (and (and (< 1 6) (on ?s ?s) ) (and (and (in_motion ?s ?p) (and (and (in_motion ?j ?p) (not (in_motion ?s) ) ) (< (distance ?s 0) 2) ) ) (not (not (in_motion ?p ?s) ) ) (in_motion ?j) (same_type ?s) ) ) ) ) ) (on ?s) ) (not (on desk) ) ) (and (and (and (not (not (on ?s) ) ) (not (or (and (in_motion ?s ?s) (on ?s ?p) ) (not (in ?s) ) (or (not (in_motion ?s ?s) ) (in_motion ?p ?p ?s) ) ) ) ) ) (not (agent_holds back) ) ) ) )
            (once (agent_holds door) )
            (hold (agent_holds ?s) )
          )
        )
      )
    )
    (preference preference2
      (exists (?i - doggie_bed)
        (then
          (once (not (in_motion ?i) ) )
          (once (rug_color_under ?i) )
          (once (adjacent ?i) )
          (once (exists (?k - hexagonal_bin) (agent_holds ?i ?k) ) )
          (once (agent_holds ?i) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 10 (count-once-per-external-objects preference2:dodgeball) )
  )
)
(:scoring
  (count preference2:pink_dodgeball)
)
)


(define (game game-id-110) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (= 3 (distance desk room_center))
  )
)
(:constraints
  (and
    (forall (?d - block ?r - cylindrical_block)
      (and
        (preference preference1
          (then
            (hold (and (agent_holds ?r ?r) (in_motion ?r ?r) ) )
            (once (agent_holds ?r) )
            (hold (and (in floor upright) (not (not (in_motion ?r) ) ) ) )
          )
        )
      )
    )
    (preference preference2
      (exists (?n - ball)
        (exists (?y - cylindrical_block)
          (exists (?w - ball ?r - (either basketball golfball))
            (then
              (once (agent_holds pink ?y) )
              (once (exists (?h - game_object) (and (agent_holds ?n) (not (not (not (not (and (agent_holds ?y) ) ) ) ) ) ) ) )
              (once (agent_holds ?y) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (- (* (count preference1:yellow) (count preference1:blue_dodgeball) )
      )
      3
    )
    (>= 4 3 )
  )
)
(:scoring
  (- (* 7 4 )
  )
)
)


(define (game game-id-111) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (and
        (and
          (game-optional
            (not
              (in_motion ?xxx)
            )
          )
        )
      )
    )
    (game-conserved
      (< (distance 0 ?xxx) (distance ?xxx desk))
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - hexagonal_bin ?o - cylindrical_block)
        (then
          (hold-while (and (in_motion ?o ?o) (and (in ?o) (not (< 1 1) ) ) (adjacent floor) ) (not (and (on ?o) (and (not (agent_holds bed) ) (in_motion ?o) ) (not (agent_holds bed) ) (not (on front) ) ) ) )
          (hold (not (adjacent ?o) ) )
          (any)
        )
      )
    )
    (preference preference2
      (then
        (hold (not (< 10 8) ) )
        (hold (not (same_color ?xxx) ) )
        (hold (and (and (on ?xxx ?xxx) (and (not (agent_holds ?xxx ?xxx) ) ) ) (and (on ?xxx) (and (in_motion door agent) (not (not (in_motion ?xxx ?xxx) ) ) ) (agent_holds ?xxx) (not (and (in_motion ?xxx) (agent_holds ?xxx) ) ) (agent_holds ?xxx) (not (agent_holds ?xxx) ) ) (not (exists (?x - game_object) (and (not (in ?x) ) (and (agent_holds ?x ?x) (agent_holds ?x agent) ) ) ) ) (not (and (not (and (agent_holds ?xxx ?xxx) (agent_holds ?xxx ?xxx) ) ) (< 7 1) ) ) ) )
      )
    )
    (forall (?a - cube_block)
      (and
        (preference preference3
          (at-end
            (in_motion ?a)
          )
        )
      )
    )
    (forall (?i - pillow ?m - dodgeball)
      (and
        (preference preference4
          (then
            (once (touch ?m ?m) )
            (once (adjacent ?m) )
            (hold-while (touch ?m tan) (not (= 1 1) ) )
          )
        )
        (preference preference5
          (exists (?e - ball)
            (exists (?n - cube_block)
              (exists (?d - book ?q - (either triangular_ramp teddy_bear))
                (exists (?k - hexagonal_bin ?v - hexagonal_bin ?f - (either main_light_switch blue_cube_block) ?i - desk_shelf)
                  (exists (?t - game_object)
                    (exists (?g - (either triangular_ramp dodgeball rug) ?z - block ?l - red_dodgeball ?l ?c - triangular_ramp ?b - dodgeball)
                      (then
                        (once (same_type agent) )
                        (once (in_motion ?i) )
                        (once (agent_holds ?m) )
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
  (= (total-time) (* (* (- (count preference1:rug) )
        (+ (* (count preference4:dodgeball) (count preference1:purple:purple) )
          (= 10 20 )
          (count preference3:pyramid_block)
          (count preference3:dodgeball)
        )
        (count-once-per-external-objects preference4:pink_dodgeball)
      )
      1
      (+ 5 (count-once-per-objects preference2:basketball) )
      (count-measure preference3:basketball:purple)
      (* (count-total preference1:beachball:tall_cylindrical_block) (count preference2:dodgeball) (count-increasing-measure preference3:blue_dodgeball) )
      (= 3 (count preference2:beachball) )
    )
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id-112) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (on ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - (either bridge_block wall))
        (exists (?u - ball ?n - hexagonal_bin ?q - dodgeball)
          (exists (?r - (either doggie_bed laptop))
            (exists (?i - building)
              (then
                (once (agent_holds back) )
                (once (and (agent_holds ?o) (in ?q ?q) ) )
                (any)
              )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?a - (either dodgeball wall))
        (exists (?p - dodgeball ?b - block ?f ?l - (either pyramid_block cylindrical_block top_drawer cube_block basketball mug triangle_block))
          (then
            (once (not (agent_holds agent rug) ) )
            (hold-while (agent_holds ?l desk) (rug_color_under ?f ?l) (not (and (exists (?g ?h - hexagonal_bin) (in_motion ?a ?l) ) (on ?f) ) ) )
            (once (not (and (and (not (in_motion ?f) ) (not (in_motion ?f agent) ) ) (touch ?f ?l) ) ) )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (not (and (agent_holds ?xxx) (in ?xxx) ) ) )
        (hold (agent_holds floor) )
        (once (same_color agent desk) )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference2:dodgeball) (count preference1:beachball:red) )
    (<= 3 (+ (- (count-once-per-objects preference3:blue_pyramid_block:red) )
        (count preference3:yellow)
      )
    )
  )
)
(:scoring
  (* (* (count preference1:golfball:blue_dodgeball:blue) (>= 3 7 )
    )
    (count-once-per-objects preference2:pink_dodgeball:dodgeball)
  )
)
)


(define (game game-id-113) (:domain few-objects-room-v1)
(:setup
  (and
    (not
      (exists (?c ?v ?t ?u ?j ?p - hexagonal_bin)
        (and
          (exists (?g - dodgeball)
            (forall (?q - (either alarm_clock doggie_bed))
              (and
                (exists (?z ?b ?l ?k ?h - red_pyramid_block)
                  (game-optional
                    (agent_holds ?q ?q)
                  )
                )
                (exists (?n - hexagonal_bin ?x - dodgeball)
                  (game-conserved
                    (on ?p)
                  )
                )
                (forall (?i - hexagonal_bin)
                  (game-optional
                    (and
                      (on ?c ?c)
                      (and
                        (agent_holds pink_dodgeball rug ?j)
                        (on ?u)
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
    (not
      (forall (?a ?g - doggie_bed ?z - hexagonal_bin)
        (and
          (and
            (exists (?l - curved_wooden_ramp ?n - building)
              (and
                (exists (?a - hexagonal_bin)
                  (and
                    (forall (?m - (either cube_block beachball cube_block))
                      (game-conserved
                        (and
                          (on ?z)
                          (not
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
    (and
      (exists (?n - (either bridge_block flat_block beachball tall_cylindrical_block key_chain cylindrical_block dodgeball))
        (and
          (forall (?f ?j ?k - blue_cube_block)
            (game-conserved
              (not
                (touch ?f ?j)
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
    (forall (?v - hexagonal_bin)
      (and
        (preference preference1
          (then
            (hold (< 3 1) )
          )
        )
      )
    )
    (preference preference2
      (exists (?g - hexagonal_bin ?x ?v ?o ?i ?t ?g - hexagonal_bin ?b - cube_block)
        (exists (?s - hexagonal_bin)
          (exists (?d - cube_block)
            (exists (?l - hexagonal_bin)
              (then
                (once (in_motion ?s) )
                (once (faces ?d) )
                (hold-while (agent_holds ?s ?b) (in ?b bed) (in_motion agent ?d) (on ?l bed) )
              )
            )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (not (in_motion ?xxx) ) )
        (once (in_motion ?xxx ?xxx) )
        (hold-while (not (on green_golfball ?xxx) ) (and (and (agent_holds upside_down) (on ?xxx) ) (agent_holds agent) (on bed ?xxx) ) )
      )
    )
  )
)
(:terminal
  (< (* (+ (and (count-once-per-objects preference2:purple) 300 ) (* (count preference3:purple) )
      )
      300
    )
    (count-once-per-objects preference1:blue_cube_block)
  )
)
(:scoring
  (= (count-once-per-objects preference3:dodgeball:dodgeball) (* 3 (count-overlapping preference3:dodgeball) )
    (count preference2:pink)
  )
)
)


(define (game game-id-114) (:domain many-objects-room-v1)
(:setup
  (forall (?l - (either dodgeball laptop))
    (exists (?v - hexagonal_bin ?d - shelf ?y - game_object)
      (and
        (exists (?p - hexagonal_bin ?u - block)
          (exists (?b - hexagonal_bin ?i - hexagonal_bin)
            (exists (?g ?n ?s - (either side_table cylindrical_block) ?b - (either basketball desktop))
              (game-conserved
                (exists (?v - block)
                  (in ?y)
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
      (exists (?c - hexagonal_bin ?j - doggie_bed ?x - dodgeball)
        (then
          (once (agent_holds ?x) )
          (once (in_motion ?x ?x) )
          (once (in ?x) )
        )
      )
    )
    (preference preference2
      (then
        (once (not (not (and (in ?xxx) (not (and (< 1 (distance 9 ?xxx)) (in_motion pink_dodgeball ?xxx) ) ) (exists (?p - (either doggie_bed yellow yellow_cube_block)) (on ?p desk) ) ) ) ) )
        (hold (>= 0.5 (distance desk back)) )
        (once-measure (not (in ?xxx ?xxx) ) (distance door 9) )
      )
    )
    (forall (?u ?a - ball ?s - curved_wooden_ramp)
      (and
        (preference preference3
          (exists (?p - hexagonal_bin)
            (then
              (hold (not (not (agent_holds ?p agent) ) ) )
              (once (and (in_motion ?p) ) )
              (hold (equal_z_position agent) )
            )
          )
        )
        (preference preference4
          (then
            (once (in ?s) )
            (once (not (exists (?o - hexagonal_bin) (agent_holds ?o ?o) ) ) )
            (hold (and (in_motion front) (agent_holds pink_dodgeball ?s) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:yellow_cube_block) (count-once preference1:red:red) )
)
(:scoring
  10
)
)


(define (game game-id-115) (:domain many-objects-room-v1)
(:setup
  (forall (?n - blue_pyramid_block ?x - doggie_bed)
    (forall (?z - hexagonal_bin)
      (game-conserved
        (agent_holds ?z ?z)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - shelf)
        (exists (?d - ball)
          (exists (?g - cylindrical_block)
            (exists (?k - teddy_bear)
              (exists (?s - (either mug cube_block pyramid_block) ?w - (either yellow_cube_block laptop hexagonal_bin dodgeball lamp cd cylindrical_block) ?t - doggie_bed)
                (exists (?h - hexagonal_bin)
                  (exists (?y - hexagonal_bin)
                    (then
                      (hold-while (touch ?d ?b) (not (forall (?l - cube_block ?e - hexagonal_bin) (not (agent_holds ?d ?h) ) ) ) (on ?b floor) )
                      (hold (is_setup_object ?t agent) )
                      (once (adjacent ?h) )
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
  (>= (count-once preference1:beachball) 8 )
)
(:scoring
  (* 3 (* (* 5 (count-once-per-objects preference1:green) )
      (count preference1:hexagonal_bin)
    )
  )
)
)


(define (game game-id-116) (:domain medium-objects-room-v1)
(:setup
  (forall (?f - hexagonal_bin ?p - dodgeball)
    (exists (?y - hexagonal_bin)
      (exists (?z - (either desktop cellphone))
        (game-conserved
          (game_over ?y)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - hexagonal_bin)
        (exists (?g - shelf ?w - building)
          (then
            (once (and (agent_holds ?s agent) (same_color north_wall ?s) (on ?s) ) )
            (once (in_motion ?s ?s) )
            (hold-while (agent_holds ?s) (and (rug_color_under blue) (adjacent_side ?w) ) )
            (once (not (adjacent ?w) ) )
          )
        )
      )
    )
    (forall (?b ?l ?d ?i - game_object)
      (and
        (preference preference2
          (then
            (once (and (in_motion bed agent) (not (or (not (adjacent ?i) ) (and (touch ?b) (< 1 (distance ?l)) ) ) ) ) )
            (once-measure (agent_holds ?l ?l) (distance ) )
            (hold-while (adjacent ?d desk) (not (and (and (and (not (agent_holds ?l) ) (on ?i) ) (touch ?l ?b) ) (and (forall (?n - hexagonal_bin) (or (not (agent_holds rug) ) (or (agent_holds ?b) (and (rug_color_under ?n) (= (x_position desk) (x_position ?n room_center)) ) ) (and (and (agent_holds ?n) (in_motion east_sliding_door) ) (in_motion ?n) (agent_holds ?l ?l) (and (broken ?b ?l) (or (and (in_motion ?i) (in ?n) ) (and (and (agent_holds ?d) (agent_holds ?i) (agent_holds ?i) ) (and (not (on upright pink_dodgeball) ) (on agent ?l) ) (not (and (agent_holds agent upright) (not (same_color ?d ?b) ) ) ) ) ) ) (and (touch ?d) (opposite ?i ?d) ) (and (in_motion ?n ?d) (not (on ?n) ) ) ) ) ) (> 1 1) ) (and (agent_holds ?i ?d) ) ) ) )
          )
        )
        (preference preference3
          (then
            (hold (> 8 (distance 6 room_center)) )
            (once (agent_holds ?i) )
            (hold-while (on ?d) (on ?l) )
          )
        )
      )
    )
    (preference preference4
      (at-end
        (and
          (in_motion ?xxx)
          (in ?xxx)
        )
      )
    )
    (preference preference5
      (then
        (once (adjacent ?xxx desk) )
        (hold (adjacent ?xxx ?xxx) )
        (once (on rug bed) )
      )
    )
    (preference preference6
      (exists (?k - chair)
        (then
          (once (on ?k) )
          (once (agent_holds ?k) )
          (once (not (in_motion ?k ?k) ) )
          (hold (on ?k) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-external-objects preference5:dodgeball) (and (count-once preference4:pink) ) )
)
(:scoring
  (count-measure preference1:dodgeball)
)
)


(define (game game-id-117) (:domain few-objects-room-v1)
(:setup
  (forall (?t - wall ?a - hexagonal_bin ?q - hexagonal_bin)
    (forall (?f - game_object)
      (exists (?x - teddy_bear)
        (forall (?u - color)
          (exists (?g ?s ?d ?k ?z ?i - ball ?w - dodgeball ?e - hexagonal_bin ?p - pillow)
            (game-conserved
              (on ?u)
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
        (once (and (not (< (distance desk bed) (distance agent ?xxx)) ) (not (not (in_motion ?xxx) ) ) ) )
        (once (adjacent ?xxx) )
        (hold (in_motion ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 3 (* (count preference1:golfball) 10 (* (count preference1:beachball:green:beachball) 50 5 (* 2 (count-once-per-objects preference1:yellow) 50 )
      )
      (total-score)
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-118) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (not
        (not
          (and
            (on ?xxx)
            (and
              (and
                (in_motion ?xxx)
                (agent_holds ?xxx ?xxx)
                (not
                  (in_motion ?xxx ?xxx)
                )
              )
              (not
                (in_motion ?xxx ?xxx)
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
      (exists (?t - hexagonal_bin ?o - game_object)
        (exists (?k - game_object)
          (exists (?u ?f - hexagonal_bin)
            (exists (?q - game_object)
              (exists (?s ?c - red_dodgeball)
                (exists (?g - ball ?h - game_object)
                  (exists (?p - green_triangular_ramp)
                    (then
                      (hold (and (in ?o) (on ?s ?f) ) )
                      (once (adjacent upright ?s) )
                      (once (not (agent_holds ?f ?f) ) )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
    (forall (?y - ball ?o - dodgeball)
      (and
        (preference preference2
          (then
            (once (< 10 2) )
            (hold (in_motion desk) )
            (once (in_motion ?o agent) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:yellow) (>= (* (* (count-once-per-objects preference1:dodgeball:red) 0 )
        (+ 5 (* (count preference2:dodgeball) 10 )
        )
      )
      (count-once-per-objects preference1)
    )
  )
)
(:scoring
  (count-once-per-objects preference1:dodgeball:beachball)
)
)


(define (game game-id-119) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (not
      (adjacent ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?y - drawer ?x - hexagonal_bin)
      (and
        (preference preference1
          (exists (?r - hexagonal_bin)
            (exists (?c - hexagonal_bin)
              (at-end
                (on ?x)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:basketball) (count-increasing-measure preference1:basketball) )
)
(:scoring
  3
)
)


(define (game game-id-120) (:domain few-objects-room-v1)
(:setup
  (or
    (and
      (game-optional
        (in_motion ?xxx)
      )
      (and
        (forall (?i - (either red pyramid_block laptop))
          (game-conserved
            (not
              (agent_holds agent)
            )
          )
        )
      )
    )
    (and
      (game-conserved
        (in ?xxx)
      )
      (and
        (game-conserved
          (same_color ?xxx bed)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?u - chair)
      (and
        (preference preference1
          (then
            (once (not (not (and (and (and (or (agent_holds ?u) (adjacent ?u door) ) (on ?u ?u) (not (not (adjacent ?u ?u) ) ) ) (not (and (and (opposite bed) (>= 1 4) (in_motion ?u ?u) (agent_holds agent) ) (forall (?f - hexagonal_bin ?y - hexagonal_bin) (exists (?i - game_object) (not (in ?i ?u) ) ) ) ) ) (and (not (exists (?w - (either alarm_clock rug)) (agent_holds ?u ?u) ) ) (and (and (same_type ?u) (agent_holds ?u ?u) ) (agent_holds ?u front) (on ?u) ) ) (object_orientation ?u) (on ?u ?u) (not (on ?u right) ) (and (not (in_motion ?u) ) (agent_holds desk) ) ) (and (agent_holds ?u ?u) (agent_holds ?u) ) ) ) ) )
            (hold (game_over ?u ?u) )
            (once (not (in_motion ?u) ) )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (touch ?xxx upright) )
        (once (in_motion ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 25 1 )
)
(:scoring
  (* (count-once-per-objects preference2:pyramid_block) 6 )
)
)


(define (game game-id-121) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-optional
          (not
            (or
              (touch ?xxx)
            )
          )
        )
      )
    )
    (and
      (and
        (game-conserved
          (not
            (not
              (and
                (in_motion rug rug)
                (in ?xxx ?xxx)
              )
            )
          )
        )
        (game-optional
          (and
            (agent_holds ?xxx ?xxx)
            (opposite ?xxx)
          )
        )
        (game-optional
          (or
            (in ?xxx)
            (touch ?xxx ?xxx)
          )
        )
      )
    )
    (and
      (and
        (exists (?f - (either pencil golfball))
          (and
            (forall (?l - hexagonal_bin ?e ?i ?o - game_object)
              (forall (?n - building)
                (game-conserved
                  (on ?o tan)
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
        (once (agent_holds ?xxx) )
        (once (on ?xxx) )
        (once (and (and (in_motion blue) (not (or (> 1 (distance desk ?xxx)) (and (adjacent top_drawer floor) (< 1 3) ) ) ) ) (not (and (in ?xxx ?xxx) (in_motion ?xxx ?xxx) ) ) ) )
      )
    )
    (preference preference2
      (then
        (hold (and (and (touch ?xxx agent) (in_motion floor agent) ) (forall (?z - doggie_bed) (not (on ?z) ) ) (and (in_motion ?xxx) (on ?xxx ?xxx) ) ) )
        (hold (in_motion ?xxx) )
        (once (adjacent ?xxx upright) )
      )
    )
  )
)
(:terminal
  (or
    (>= (+ (count preference2:dodgeball:basketball) (* 5 (* 10 )
        )
        10
        40
        8
        (+ (= (count preference2:blue_dodgeball:red) (count-once-per-external-objects preference1:pink) )
          (count preference1:dodgeball)
        )
        (count preference2:doggie_bed)
      )
      (count-measure preference1:dodgeball)
    )
    (> 50 (count preference1:dodgeball) )
  )
)
(:scoring
  (count preference1:dodgeball)
)
)


(define (game game-id-122) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (in ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - hexagonal_bin)
        (exists (?r - (either pencil flat_block))
          (exists (?y - dodgeball)
            (exists (?p - building)
              (exists (?u - hexagonal_bin ?w - dodgeball ?v - hexagonal_bin ?z - dodgeball)
                (at-end
                  (in_motion desk ?o ?y)
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?j - dodgeball)
        (exists (?v - doggie_bed ?g - wall)
          (then
            (hold (not (not (not (or (not (and (and (< 3 (distance front_left_corner ?j)) (agent_holds ?g ?g) (agent_holds ?g) (on ?j ?g) ) (< (distance 8 room_center) 9) ) ) ) ) ) ) )
            (once (in ?g ?j) )
            (hold (opposite bed ?g) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:beachball) 30 )
)
(:scoring
  (= (* (count preference2:dodgeball) (+ (count preference1:golfball) (* (+ (* (count-once-per-objects preference2:doggie_bed) (count-once-per-objects preference1:hexagonal_bin) )
            (count-overlapping preference2:dodgeball)
            5
          )
          (count preference1:pyramid_block)
          (count-once-per-objects preference1)
          16
        )
        4
        (* (+ 4 (external-forall-maximize (count-once-per-objects preference2:basketball) ) )
          (* 1 (* 2 1 )
          )
        )
      )
    )
  )
)
)


(define (game game-id-123) (:domain few-objects-room-v1)
(:setup
  (forall (?b - rug)
    (game-conserved
      (adjacent desk)
    )
  )
)
(:constraints
  (and
    (forall (?h ?u ?w ?q ?v - hexagonal_bin)
      (and
        (preference preference1
          (exists (?o - hexagonal_bin ?x - hexagonal_bin ?x - teddy_bear)
            (then
              (once (agent_holds bed ?u) )
              (once (on ?h ?x) )
              (hold (not (forall (?y - building) (in_motion ?u ?x) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* 9 (or (* (count-once-per-objects preference1:pink_dodgeball:basketball) (external-forall-maximize 5 ) (count preference1:blue_dodgeball:yellow_cube_block) (* 8 (count preference1:red) )
            5
            (+ 20 15 )
          )
        )
      )
      (* 3 (external-forall-maximize 10 ) )
    )
    (not
      (>= (/ (count preference1:beachball) (* 3 (total-score) )
        )
        (count-once preference1:orange)
      )
    )
    (= 10 4 )
  )
)
(:scoring
  (+ (external-forall-maximize (* (+ 9 (count-same-positions preference1:top_drawer:blue_dodgeball) )
        (* (count-once-per-objects preference1:golfball) (count preference1:blue_dodgeball) )
        (count preference1:dodgeball)
      )
    )
    (external-forall-maximize
      (* 3 (+ 10 (count-measure preference1:dodgeball:blue_pyramid_block) (count preference1:dodgeball:dodgeball:doggie_bed) )
        (count preference1:pink)
      )
    )
  )
)
)


(define (game game-id-124) (:domain few-objects-room-v1)
(:setup
  (forall (?k - pyramid_block)
    (and
      (and
        (and
          (game-conserved
            (agent_holds ?k ?k)
          )
        )
        (game-optional
          (not
            (and
              (agent_holds ?k)
              (and
                (in agent pillow ?k)
                (in_motion ?k)
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
      (exists (?l - hexagonal_bin)
        (then
          (once (exists (?k - curved_wooden_ramp) (agent_holds ?l) ) )
          (once (is_setup_object ?l ?l) )
          (once (touch ?l) )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference1:hexagonal_bin:yellow) (count preference1:blue_dodgeball) )
)
(:scoring
  8
)
)


(define (game game-id-125) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - hexagonal_bin)
        (then
          (once (not (in ?c ?c) ) )
          (once (and (not (in ?c ?c) ) (in_motion bed) ) )
          (hold (and (not (between ?c ?c) ) (not (not (and (and (agent_holds ?c ?c) (not (above ?c ?c ?c) ) ) (not (adjacent ?c ?c) ) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (not
        (or
          (>= (count-once-per-objects preference1:pink_dodgeball) (count preference1:dodgeball) )
          (>= 10 15 )
        )
      )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-126) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (exists (?l - dodgeball)
        (not
          (game-conserved
            (equal_x_position green ?l)
          )
        )
      )
      (game-conserved
        (agent_holds ?xxx ?xxx)
      )
      (forall (?u - dodgeball ?p - (either key_chain wall))
        (game-optional
          (agent_holds agent)
        )
      )
    )
    (game-conserved
      (in_motion upright)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (in ?xxx desk) )
        (once (in_motion agent ?xxx) )
        (hold (and (in_motion ?xxx) (adjacent ?xxx) ) )
        (hold (not (in_motion ?xxx) ) )
        (once (and (not (agent_holds ?xxx) ) (in_motion ?xxx) (not (agent_holds ?xxx) ) ) )
      )
    )
    (preference preference2
      (then
        (once (exists (?u - hexagonal_bin) (agent_holds ?u ?u) ) )
        (once (forall (?u - wall) (or (not (< 2 (distance ?u ?u)) ) (agent_holds ?u ?u) ) ) )
        (hold (and (< (distance 8 agent) 6) (and (same_object ?xxx ?xxx) (touch ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (>= 3 (= (- 3 )
    )
  )
)
(:scoring
  (* (count preference2:dodgeball) (count preference1) (count preference2:blue) )
)
)


(define (game game-id-127) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (on ?xxx)
  )
)
(:constraints
  (and
    (forall (?g - watch)
      (and
        (preference preference1
          (at-end
            (adjacent ?g)
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
        (and
          (>= (count-once-per-objects preference1:pink) (count-once preference1:basketball:wall) )
          (< (count preference1:dodgeball) (total-time) )
          (and
            (<= (* (+ (count preference1:side_table) (+ (* (count-once-per-objects preference1:pink_dodgeball) (* 5 2 )
                    )
                    (count-once preference1:pink)
                  )
                  (- (count preference1:beachball) )
                )
                (external-forall-maximize
                  (* 5 (* 4 (+ (* (+ 4 (count preference1:dodgeball) )
                          (count-once-per-external-objects preference1)
                        )
                        (= 8 (* (total-score) 2 (count-once preference1:blue_dodgeball) )
                        )
                      )
                      2
                      (and
                        (count preference1:beachball:dodgeball)
                        (* 15 (count preference1:pink:dodgeball) )
                        3
                      )
                    )
                  )
                )
              )
              (count preference1:beachball:basketball)
            )
            (> (* (count preference1:beachball) 10 (- 10 (* (count-same-positions preference1:yellow) (* (+ 6 (- (count-once-per-objects preference1:dodgeball) )
                      )
                      (count preference1:basketball)
                      (count-once-per-objects preference1:yellow_pyramid_block)
                    )
                  )
                )
                (* 2 3 (count-once preference1:purple:blue_dodgeball) )
                (count-once-per-objects preference1:blue_pyramid_block)
                4
              )
              (+ (< (count preference1:beachball) 5 )
                (total-time)
              )
            )
            (> (* (= (+ (- (- (count preference1:basketball:dodgeball:pink) 300 ) )
                    2
                  )
                  (count-shortest preference1:red:golfball)
                )
                (+ (count preference1:brown:cylindrical_block) 3 )
                (< 5 9 )
              )
              3
            )
          )
        )
        (> (external-forall-maximize 1 ) 100 )
      )
      (<= (* (total-score) 3 )
        (* (count preference1:brown) 3 )
      )
    )
    (> 2 (count-once preference1:dodgeball) )
  )
)
(:scoring
  2
)
)


(define (game game-id-128) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?y - cube_block)
      (exists (?d - block ?s - hexagonal_bin ?m - ball)
        (forall (?u - hexagonal_bin ?h - hexagonal_bin ?b - ball)
          (and
            (and
              (and
                (or
                  (and
                    (exists (?g - building)
                      (game-conserved
                        (and
                          (= (distance agent ?y))
                          (on ?y ?b)
                          (not
                            (agent_holds ?g ?b ?g)
                          )
                        )
                      )
                    )
                  )
                )
                (exists (?u - hexagonal_bin)
                  (game-conserved
                    (and
                      (in_motion ?u)
                      (not
                        (and
                          (on rug)
                          (agent_holds ?b ?u)
                        )
                      )
                    )
                  )
                )
                (game-conserved
                  (and
                    (in ?b ?m)
                    (same_color ?y)
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
      (exists (?d - cube_block)
        (then
          (once (or (< 1 1) (agent_holds ?d) (= 4 (distance 6 6)) ) )
          (hold (and (in_motion desk) (on ?d ?d) ) )
          (once (in green_golfball) )
        )
      )
    )
    (preference preference2
      (exists (?b - game_object ?s - dodgeball)
        (exists (?l - ball)
          (exists (?r - teddy_bear)
            (then
              (once (and (not (on ?l ?l) ) (and (and (agent_holds ?l) (not (not (touch ?l) ) ) ) (in_motion right) ) ) )
              (once (not (on ?l) ) )
              (once (on ?l ?l) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (count preference1:cylindrical_block:hexagonal_bin:yellow) )
)
(:scoring
  2
)
)


(define (game game-id-129) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (in agent)
      )
    )
  )
)
(:constraints
  (and
    (forall (?g - ball)
      (and
        (preference preference1
          (exists (?k - dodgeball)
            (then
              (hold-while (on ?g) (and (object_orientation ?k) (in_motion pink_dodgeball) (agent_holds block) ) )
              (once (not (not (in ?g) ) ) )
              (once (and (agent_holds ?k) (same_color front) ) )
            )
          )
        )
      )
    )
    (forall (?n - block)
      (and
        (preference preference2
          (exists (?x - (either alarm_clock bridge_block))
            (then
              (once (in ?x) )
              (once (and (agent_holds ?x ?n) (in_motion ?n ?n) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (* 9 (- 0 )
      )
      (count-once-per-objects preference2:yellow)
    )
    (> (total-score) 1 )
  )
)
(:scoring
  7
)
)


(define (game game-id-130) (:domain few-objects-room-v1)
(:setup
  (and
    (game-optional
      (touch ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (in_motion ?xxx pink_dodgeball) (and (touch ?xxx ?xxx) (and (same_object ?xxx ?xxx) (agent_holds ?xxx) (opposite ?xxx ?xxx) (rug_color_under ?xxx) ) ) (and (in_motion ?xxx ?xxx) (or (agent_holds agent bed) (not (not (not (in_motion ?xxx ?xxx) ) ) ) ) ) )
        (once (not (agent_holds ?xxx ?xxx) ) )
        (hold-while (and (agent_holds desk) (agent_holds ?xxx ?xxx) ) (in_motion ?xxx ?xxx) (agent_holds rug) )
      )
    )
  )
)
(:terminal
  (>= (count preference1:basketball) (count preference1:pink) )
)
(:scoring
  (count preference1:blue_dodgeball)
)
)


(define (game game-id-131) (:domain few-objects-room-v1)
(:setup
  (forall (?e - cube_block)
    (exists (?o - hexagonal_bin ?c - building)
      (game-optional
        (on floor ?e)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z ?i - ball)
        (exists (?y - (either golfball dodgeball desktop))
          (exists (?b - chair)
            (at-end
              (equal_z_position ?z ?z)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (< 2 (count-shortest preference1:tan:golfball) )
    (or
      (and
        (>= 2 (count preference1:red:dodgeball) )
        (and
          (>= 0.5 (count preference1:tall_cylindrical_block) )
        )
      )
      (>= (external-forall-minimize 5 ) 18 )
    )
    (>= 2 (+ (* (* 50 (count preference1:pink_dodgeball) )
          (count-unique-positions preference1:dodgeball:tan)
        )
        (* 180 (count-once-per-objects preference1:dodgeball:dodgeball:basketball) )
      )
    )
  )
)
(:scoring
  (* (* (count-unique-positions preference1:pyramid_block:pink_dodgeball) (count-once-per-external-objects preference1:yellow:pink) )
    3
  )
)
)


(define (game game-id-132) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (adjacent_side ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - hexagonal_bin)
        (then
          (hold-while (in_motion ?n) (and (< 6 (distance ?n ?n)) (in ?n ?n) ) )
          (once (not (and (in_motion ?n) (adjacent ?n) ) ) )
          (once (on ?n ?n) )
        )
      )
    )
    (forall (?j - dodgeball)
      (and
        (preference preference2
          (then
            (hold-for 7 (and (in_motion ?j ?j) (in_motion ?j) ) )
            (once (on ?j ?j) )
            (once (and (same_color ?j) (and (agent_holds ?j ?j) (agent_holds ?j) ) (in_motion ?j ?j) (on ?j) (adjacent ?j) (adjacent ?j ?j) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:dodgeball) (not 10 ) )
)
(:scoring
  10
)
)


(define (game game-id-133) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (and
      (agent_holds ?xxx ?xxx)
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - beachball)
        (exists (?c - dodgeball)
          (then
            (once (and (in_motion bed ?c) (touch ?c) ) )
            (once (on ?q ?q) )
            (once (forall (?o - hexagonal_bin) (in_motion ?o back) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- (- 2 )
    )
    (* (+ (external-forall-maximize (external-forall-minimize (* (count-longest preference1:red) 6 )
          )
        )
        2
        (count preference1:blue_dodgeball)
      )
      (count preference1:golfball:wall)
    )
  )
)
(:scoring
  (* (count preference1:cylindrical_block) 3 )
)
)


(define (game game-id-134) (:domain few-objects-room-v1)
(:setup
  (forall (?b - desktop ?e - hexagonal_bin)
    (or
      (game-conserved
        (not
          (in_motion ?e)
        )
      )
      (game-conserved
        (not
          (in_motion ?e)
        )
      )
      (and
        (forall (?h - ball)
          (forall (?m - hexagonal_bin)
            (exists (?f - (either lamp dodgeball dodgeball cd cube_block dodgeball dodgeball))
              (and
                (forall (?y - (either bridge_block))
                  (exists (?d - bridge_block ?o - hexagonal_bin ?k - game_object)
                    (game-optional
                      (adjacent ?h)
                    )
                  )
                )
                (exists (?u - cylindrical_block)
                  (game-conserved
                    (in_motion right)
                  )
                )
              )
            )
          )
        )
        (exists (?i - building ?g - (either cube_block desktop) ?f - hexagonal_bin)
          (exists (?t - triangular_ramp ?y - triangular_ramp ?y - building ?a - hexagonal_bin)
            (exists (?b - hexagonal_bin)
              (game-conserved
                (on ?e ?b)
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
      (exists (?c - hexagonal_bin)
        (exists (?d - ball)
          (exists (?f ?g - pillow)
            (exists (?n - hexagonal_bin)
              (then
                (hold-while (in_motion ?n ?f) (and (agent_holds sideways) (exists (?y - triangular_ramp ?t - game_object) (and (and (in ?f) (in_motion floor) ) (not (agent_holds ?d) ) ) ) (agent_holds rug ?c) (or (or (in ?c ?d) (touch desk ?n) ) ) (exists (?z - game_object) (agent_holds ?g ?d) ) (and (agent_holds ?g) (agent_holds ?f ?g) ) ) )
                (hold (and (equal_z_position ?c) (not (same_object ?c ?f) ) ) )
                (once (in ) )
                (hold (= 1 (distance desk agent)) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (+ 30 (count-longest preference1:wall:tan) )
    0
  )
)
(:scoring
  (count preference1:green)
)
)


(define (game game-id-135) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (not
          (in_motion ?xxx)
        )
      )
    )
    (and
      (exists (?w - red_dodgeball)
        (game-optional
          (forall (?m - hexagonal_bin)
            (forall (?z - wall)
              (not
                (not
                  (is_setup_object ?m)
                )
              )
            )
          )
        )
      )
      (exists (?r - curved_wooden_ramp)
        (exists (?h - block)
          (or
            (game-optional
              (adjacent ?h)
            )
            (and
              (and
                (and
                  (and
                    (game-conserved
                      (and
                        (not
                          (= (distance 10 ?h) (distance ?r agent desk))
                        )
                        (> 7 (distance_side ?r 9))
                      )
                    )
                    (not
                      (and
                        (game-conserved
                          (adjacent_side ?r bed)
                        )
                        (and
                          (game-conserved
                            (adjacent ?h)
                          )
                        )
                        (and
                          (game-conserved
                            (in ?r)
                          )
                        )
                      )
                    )
                  )
                  (exists (?y - watch ?b ?l - book)
                    (and
                      (and
                        (exists (?f - color)
                          (game-optional
                            (on ?r ?f)
                          )
                        )
                      )
                      (and
                        (game-conserved
                          (agent_holds ?h ?r)
                        )
                        (forall (?q ?v - ball)
                          (exists (?a - chair)
                            (game-conserved
                              (same_type desk ?h)
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (or
                  (and
                    (game-conserved
                      (in_motion ?r ?h)
                    )
                    (game-optional
                      (in_motion ?h ?h)
                    )
                    (or
                      (forall (?x - teddy_bear)
                        (and
                          (exists (?q - dodgeball ?f ?s ?n ?z ?l - ball ?n - dodgeball)
                            (game-conserved
                              (agent_holds ?x ?h ?x)
                            )
                          )
                          (and
                            (forall (?c - cube_block)
                              (game-conserved
                                (not
                                  (equal_x_position ?r)
                                )
                              )
                            )
                          )
                        )
                      )
                      (exists (?t - dodgeball)
                        (game-optional
                          (and
                            (agent_holds ?t)
                            (and
                              (not
                                (in_motion ?h ?h)
                              )
                              (not
                                (in_motion ?r ?t)
                              )
                            )
                            (adjacent_side ?h)
                          )
                        )
                      )
                    )
                  )
                  (and
                    (and
                      (game-optional
                        (and
                          (in_motion ?h)
                          (not
                            (> 1 (distance ?h ?h))
                          )
                          (and
                            (agent_holds agent ?h)
                            (< 1 9)
                            (in ?h)
                          )
                        )
                      )
                      (game-conserved
                        (and
                          (and
                            (not
                              (exists (?k - doggie_bed)
                                (faces ?r rug)
                              )
                            )
                            (and
                              (game_over ?r agent)
                              (in_motion ?h)
                            )
                          )
                        )
                      )
                    )
                  )
                  (exists (?t - ball)
                    (game-conserved
                      (in_motion agent ?t)
                    )
                  )
                )
              )
              (exists (?x - dodgeball ?j - ball)
                (game-conserved
                  (on front ?j)
                )
              )
            )
            (game-conserved
              (same_color ?h)
            )
          )
        )
      )
    )
    (and
      (or
        (game-conserved
          (open west_wall)
        )
        (and
          (game-conserved
            (opposite ?xxx ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - (either alarm_clock yellow dodgeball) ?e - dodgeball)
        (exists (?l - flat_block)
          (exists (?w - chair ?q - ball)
            (then
              (hold (> (distance ?q ?l) 1) )
              (once (on ?l) )
              (once (same_color ?l ?q) )
              (once (in_motion ?q agent) )
              (hold (and (and (exists (?b - hexagonal_bin ?z - block) (and (in_motion ?l) (not (in_motion ?q) ) ) ) (agent_holds ?e) ) (not (and (not (or (and (on ?l ?l ?q) (and (agent_holds ?l ?q) (same_type ?l floor) (not (not (in brown) ) ) ) ) (in_motion agent ?q) (adjacent ?e) ) ) (and (agent_holds ?q ?e) (or (adjacent bed) (in_motion ?q) ) ) ) ) ) )
              (hold-while (and (not (not (agent_holds ?e) ) ) (or (not (in_motion ?l ?l) ) ) ) (in ?e) (in_motion ?q) (not (on agent) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (or
      (>= (count preference1:red:blue_dodgeball) (count-once preference1:yellow:golfball) )
      (> (+ 3 (total-time) )
        (total-time)
      )
    )
  )
)
(:scoring
  (count preference1:blue_cube_block)
)
)


(define (game game-id-136) (:domain few-objects-room-v1)
(:setup
  (and
    (game-optional
      (in ?xxx)
    )
    (and
      (game-conserved
        (not
          (not
            (in_motion ?xxx)
          )
        )
      )
      (game-conserved
        (same_color ?xxx ?xxx)
      )
      (and
        (exists (?f - chair)
          (game-conserved
            (= (distance desk ?f) 5)
          )
        )
        (and
          (game-conserved
            (and
              (and
                (or
                  (in_motion ?xxx)
                  (not
                    (and
                      (touch ?xxx ?xxx)
                      (and
                        (in ?xxx)
                        (not
                          (adjacent ?xxx ?xxx)
                        )
                      )
                      (not
                        (adjacent_side ?xxx ?xxx)
                      )
                    )
                  )
                )
                (not
                  (on ?xxx)
                )
              )
              (in_motion bed ?xxx)
            )
          )
        )
      )
    )
    (or
      (and
        (game-conserved
          (exists (?b - game_object ?t - building)
            (< (x_position ?t room_center) (building_size ?t room_center))
          )
        )
        (exists (?n - hexagonal_bin ?u - pyramid_block ?v - dodgeball)
          (and
            (exists (?h - game_object)
              (game-conserved
                (not
                  (on ?v)
                )
              )
            )
            (game-conserved
              (in_motion green_golfball ?v)
            )
            (and
              (game-conserved
                (agent_holds ?v)
              )
              (exists (?n - (either cellphone dodgeball))
                (game-conserved
                  (and
                    (and
                      (and
                        (on ?n)
                        (exists (?k - sliding_door ?y - ball)
                          (in_motion ?n)
                        )
                      )
                      (same_color ?v)
                    )
                    (not
                      (agent_holds ?n)
                    )
                  )
                )
              )
              (exists (?b - (either cylindrical_block cellphone))
                (game-optional
                  (in ?b ?v)
                )
              )
            )
          )
        )
        (game-conserved
          (not
            (not
              (in_motion ?xxx ?xxx)
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
        (once (exists (?a - hexagonal_bin ?k - wall) (and (and (agent_holds floor ?k) (agent_holds ?k) (on ?k ?k) (agent_holds agent) ) (and (in_motion ?k) (in ?k) ) ) ) )
        (once (and (and (and (and (in_motion ?xxx) (not (not (< 3 (distance room_center ?xxx)) ) ) ) (and (agent_holds ?xxx ?xxx) (and (agent_holds front green) (and (in_motion ?xxx ?xxx) (in ?xxx) ) ) ) (touch ?xxx) (< 6 1) ) (on ?xxx ?xxx) (rug_color_under ?xxx ?xxx) (agent_holds bed ?xxx) ) (on ?xxx ?xxx) (and (or (and (= 1) (above agent) ) (in_motion ?xxx ?xxx) ) (in_motion ?xxx ?xxx) ) ) )
        (hold-while (< (building_size ?xxx) 0) (not (and (not (not (and (and (< 1 6) (agent_holds ?xxx ?xxx ?xxx) ) (not (not (not (not (agent_holds ?xxx) ) ) ) ) ) ) ) (in_motion ?xxx) ) ) (and (equal_x_position ?xxx ?xxx) (in_motion right) ) (in ?xxx ?xxx) )
        (once (or (and (> (distance 2 ?xxx) 0.5) (in_motion ?xxx) ) (in_motion ?xxx ?xxx) ) )
      )
    )
    (preference preference2
      (exists (?s - hexagonal_bin)
        (then
          (once (in_motion bed ?s) )
          (hold-while (in_motion ?s) (agent_holds ?s) (in_motion ?s) )
          (hold (touch ?s ?s) )
          (once (not (and (forall (?e - cube_block) (on ?e ?e) ) (or (agent_holds sideways ?s) (< (distance ?s bed) 1) ) ) ) )
          (once (on ?s ?s ?s) )
        )
      )
    )
    (preference preference3
      (exists (?w - desk_shelf ?n - hexagonal_bin)
        (exists (?h - dodgeball ?e - wall)
          (then
            (hold (in ?e) )
            (once (and (or (exists (?y - ball) (not (and (agent_holds ?e upright) (exists (?z - (either pink book teddy_bear) ?z - hexagonal_bin ?p - building) (in_motion desk) ) ) ) ) ) (not (touch ?n ?e) ) (in_motion ?e agent) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference3:dodgeball) (/ (* (count preference1:basketball) (count preference1:yellow_cube_block:red) )
      (+ (count-once-per-objects preference1:basketball) (+ (+ (count preference3) (+ (total-score) )
            (count-same-positions preference2:basketball)
            (+ (count preference2:beachball:triangle_block) 2 )
            (* (count preference3:pink) (count preference3:yellow_cube_block) )
            2
          )
          5
        )
      )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-137) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (not
      (in_motion agent)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - ball)
        (exists (?d - hexagonal_bin ?n - dodgeball)
          (then
            (hold-while (on ?n) (not (and (not (in_motion desk) ) (and (not (in ?n) ) (agent_holds ?e) ) (not (touch ?n) ) ) ) )
            (once (in_motion ?e ?e) )
            (once (in_motion left ?e) )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count preference1:book) 5 )
    (> (count-once-per-objects preference1:basketball) (count preference1:top_drawer) )
  )
)
(:scoring
  (count preference1:red:basketball)
)
)


(define (game game-id-138) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (exists (?w - dodgeball)
      (and
        (and
          (and
            (not
              (adjacent ?w agent)
            )
            (and
              (on west_wall)
              (adjacent ?w)
            )
          )
          (or
            (not
              (and
                (in ?w)
                (agent_holds ?w ?w)
              )
            )
            (and
              (agent_holds agent)
              (on ?w desk)
              (not
                (and
                  (and
                    (and
                      (agent_holds ?w)
                      (not
                        (agent_holds ?w ?w)
                      )
                      (not
                        (in ?w)
                      )
                    )
                    (on rug ?w)
                  )
                  (rug_color_under ?w)
                )
              )
            )
          )
          (and
            (and
              (and
                (in ?w ?w)
                (in_motion agent)
              )
              (and
                (= (distance ?w front) 0.5)
                (and
                  (exists (?h - (either pyramid_block wall))
                    (agent_holds ?w ?w)
                  )
                  (not
                    (in_motion ?w ?w)
                  )
                )
              )
            )
          )
        )
        (and
          (forall (?t - cube_block ?g - game_object)
            (and
              (not
                (and
                  (in ?w desk)
                  (or
                    (in_motion ?w)
                    (agent_holds ?w ?w)
                  )
                )
              )
              (agent_holds ?w ?g)
            )
          )
          (in_motion ?w)
          (agent_holds ?w)
          (in_motion ?w ?w)
        )
        (in_motion ?w ?w)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - golfball)
        (at-end
          (agent_holds ?z)
        )
      )
    )
    (preference preference2
      (exists (?j - hexagonal_bin ?y - doggie_bed)
        (then
          (once (agent_holds ?y) )
          (once (exists (?o - (either basketball dodgeball flat_block book alarm_clock book pen)) (agent_holds ?y) ) )
          (once (and (in ?y) (exists (?v - cube_block) (adjacent ?v ?y) ) ) )
        )
      )
    )
    (forall (?w - dodgeball)
      (and
        (preference preference3
          (exists (?m - building)
            (exists (?e - hexagonal_bin)
              (then
                (once (adjacent ?m ?e) )
                (hold-while (not (and (and (exists (?f - wall) (faces ?f) ) (exists (?d - game_object) (and (and (not (and (in ?d) (in_motion ?m) ) ) (agent_holds ?m ?w) ) (and (not (not (agent_holds ?d) ) ) (agent_holds ?m) ) (not (in ?d) ) (and (not (forall (?z - color) (agent_holds ?m) ) ) (agent_holds ?d) (same_color ?d) ) ) ) ) (and (is_setup_object bed ?w) (in_motion door) ) (agent_holds ?e) ) ) (and (in_motion agent) (not (in_motion agent side_table) ) ) )
                (once (in ?m) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-longest preference1:basketball) (* (count-once-per-objects preference3:pink:dodgeball:rug) 6 )
  )
)
(:scoring
  8
)
)


(define (game game-id-139) (:domain medium-objects-room-v1)
(:setup
  (not
    (forall (?y - (either flat_block cube_block) ?r - (either golfball triangle_block))
      (forall (?o - dodgeball)
        (and
          (exists (?a - yellow_cube_block ?n ?b - game_object)
            (forall (?e - wall ?z ?k ?s - block ?w - cube_block)
              (and
                (game-conserved
                  (and
                    (> (distance room_center ?r) (distance desk room_center ?n))
                    (in_motion ?n)
                    (agent_holds ?n)
                    (in_motion ?w)
                  )
                )
                (game-conserved
                  (not
                    (not
                      (not
                        (in_motion ?w)
                      )
                    )
                  )
                )
                (and
                  (game-conserved
                    (in_motion ?o ?r)
                  )
                  (exists (?y - doggie_bed)
                    (exists (?a - dodgeball)
                      (forall (?c - golfball ?f - dodgeball)
                        (game-conserved
                          (in ?r)
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
    (preference preference1
      (exists (?n - ball ?e - block)
        (at-end
          (exists (?r - (either golfball yellow_cube_block) ?z - triangular_ramp ?u - wall ?n - (either laptop yellow_cube_block desktop))
            (on ?e)
          )
        )
      )
    )
    (forall (?g - ball)
      (and
        (preference preference2
          (then
            (once (in_motion upright ?g) )
            (once (touch ?g ?g) )
            (hold (and (not (not (and (and (agent_holds ?g) (touch bed) ) (and (on ?g ?g) (agent_holds ?g ?g) (and (and (agent_holds ?g ?g) (not (in_motion ?g ?g) ) ) (in front_left_corner) (agent_holds ?g) ) ) ) ) ) (and (on ?g ?g) (= (distance room_center 5) 4) ) (< (distance 8 ?g) (distance ?g ?g)) ) )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (in desk ?xxx) )
        (once (in_motion ?xxx) )
        (once (exists (?j - ball) (and (and (not (forall (?p ?o - hexagonal_bin ?o - block) (exists (?c - (either golfball golfball) ?t - hexagonal_bin) (on ?t ?o) ) ) ) (not (in_motion ?j ?j) ) ) (in_motion agent) ) ) )
      )
    )
  )
)
(:terminal
  (and
    (>= (count preference2:hexagonal_bin) (count-once-per-objects preference1:dodgeball) )
    (>= (* (count preference2:blue_cube_block:doggie_bed) (count-once-per-objects preference3:beachball) )
      (count preference1:blue_cube_block)
    )
    (or
      (>= 5 (count preference3:green) )
      (or
        (or
          (>= (count-once-per-objects preference3:yellow) 100 )
          (>= 2 (external-forall-maximize 9 ) )
          (not
            (or
              (> (* (external-forall-minimize (count preference1:beachball) ) (count-total preference3:pink) (count preference1:dodgeball) 3 (* 10 (count-once-per-objects preference3:book) )
                  (count preference2:cube_block:hexagonal_bin)
                )
                10
              )
              (> (count-same-positions preference3:dodgeball:yellow_pyramid_block) 1 )
            )
          )
        )
        (and
          (and
            (= (= (count preference3:green:pink) (count preference3:dodgeball) (count-once-per-objects preference3:golfball:beachball) )
              5
            )
            (or
              (>= (total-score) 9 )
              (>= 3 (or 2 (count-once-per-objects preference1:dodgeball) (count-once-per-objects preference2:cube_block:dodgeball) ) )
              (or
                (>= (count-once preference1:dodgeball:beachball) (count-once-per-objects preference1:alarm_clock) )
              )
            )
          )
          (not
            (= (total-score) (* (count preference2:dodgeball:cube_block) 2 )
            )
          )
          (>= (+ (* (* (* (+ (count-same-positions preference1:dodgeball) (count-overlapping preference1:blue_dodgeball:pink_dodgeball) )
                    (* (/ (count preference1:pink) (* (count preference2:pink:yellow) (count-once-per-external-objects preference1:pink_dodgeball) )
                      )
                      (count-once-per-objects preference2:hexagonal_bin:dodgeball)
                    )
                    (* (* 9 (external-forall-maximize (+ 3 (count preference2) )
                        )
                      )
                      (count preference2:beachball)
                      (count preference1:dodgeball)
                      50
                      6
                      (count preference2:bed)
                    )
                    (count-once-per-objects preference3:yellow_cube_block)
                    (/
                      (count-once-per-objects preference2:beachball)
                      (count-overlapping preference3:dodgeball:pink_dodgeball)
                    )
                    (+ (count-increasing-measure preference2:dodgeball:basketball) 3 )
                  )
                  (count-once-per-objects preference1)
                )
                (- 3 )
              )
              (* (* (* (+ 3 1 )
                    5
                  )
                  (+ (* 2 15 )
                  )
                )
                180
              )
              (* (* (* (+ (count preference2:tall_cylindrical_block) (* 5 (count preference3:dodgeball) )
                      (* (* 30 (count-once preference1:beachball:blue_dodgeball) )
                        (count-once-per-objects preference2:green:pink_dodgeball)
                      )
                      (count preference2:alarm_clock:hexagonal_bin)
                    )
                    (= (count-total preference1:hexagonal_bin) (total-score) (total-score) )
                  )
                  (total-score)
                  6
                  (* (count-measure preference2:green:basketball) (count-longest preference3:pink:golfball) )
                  (-
                    (count-once-per-objects preference1:cube_block:pyramid_block)
                    50
                  )
                  5
                )
                (external-forall-maximize
                  5
                )
                (count preference1:hexagonal_bin)
                (+ (count preference3:dodgeball) (+ (count-once-per-external-objects preference1) (* (- 3 )
                      (count-once-per-objects preference1:pink)
                    )
                    (total-score)
                    (* (count preference3:blue_cube_block:book) 10 (count preference3:hexagonal_bin:red:orange) 5 )
                    (count-once-per-objects preference3:dodgeball)
                    (count-once-per-external-objects preference1:dodgeball)
                    (* 6 (* (count-once-per-objects preference1:golfball:pyramid_block) (count-once-per-external-objects preference3:blue_pyramid_block:tan) 3 (count preference1:side_table) )
                    )
                    (>= (= (external-forall-minimize (count preference1:dodgeball) ) 0 )
                      (count preference1:golfball)
                    )
                    (count preference2:basketball:dodgeball)
                  )
                )
              )
            )
            50
          )
        )
      )
      (not
        (>= (* (+ (* (count preference1:hexagonal_bin) (- 8 )
                2
              )
              (count-once preference3:dodgeball)
            )
            (* (count preference3:bed) (* 60 (count preference3:red) )
            )
          )
          (* (count-once preference1:pink_dodgeball) 5 (* 5 10 )
          )
        )
      )
    )
  )
)
(:scoring
  (* (count-measure preference2:pink) (* (count preference2:dodgeball) (count preference3:golfball) )
  )
)
)


(define (game game-id-140) (:domain many-objects-room-v1)
(:setup
  (exists (?d - (either pyramid_block tall_cylindrical_block book))
    (exists (?p - cube_block)
      (forall (?q - dodgeball)
        (and
          (exists (?o - teddy_bear)
            (exists (?z - hexagonal_bin)
              (game-conserved
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
(:constraints
  (and
    (forall (?c - dodgeball)
      (and
        (preference preference1
          (then
            (once (in ?c ?c) )
            (once (in ball) )
            (once (and (touch desk agent) (and (and (equal_z_position ?c) (< 4 (distance room_center ?c ?c)) ) (in_motion bed) ) (in_motion top_drawer bed ?c) ) )
          )
        )
        (preference preference2
          (exists (?w - dodgeball)
            (exists (?n - block)
              (exists (?e - shelf)
                (then
                  (hold (not (in_motion ?n) ) )
                  (once (and (on bed) (exists (?t ?m ?r - hexagonal_bin) (<= (distance bed ?e) 7) ) ) )
                  (hold (not (not (in_motion ?e ?n) ) ) )
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
  (not
    (>= 3 30 )
  )
)
(:scoring
  (not
    (* 5 (count-unique-positions preference2:green) )
  )
)
)


(define (game game-id-141) (:domain many-objects-room-v1)
(:setup
  (exists (?f ?k - dodgeball)
    (game-conserved
      (< (distance 6 ?k) 1)
    )
  )
)
(:constraints
  (and
    (forall (?u - hexagonal_bin)
      (and
        (preference preference1
          (exists (?m - (either hexagonal_bin desktop))
            (exists (?i - shelf)
              (then
                (once (in_motion ?m ?m) )
                (hold (in_motion ?i) )
              )
            )
          )
        )
        (preference preference2
          (then
            (once (agent_holds ?u) )
            (hold (or (agent_holds ?u) (in_motion ?u) ) )
            (hold (and (in north_wall ?u agent) (in_motion desk) ) )
          )
        )
      )
    )
    (forall (?t - building ?b - doggie_bed)
      (and
        (preference preference3
          (then
            (hold-while (and (in_motion ?b) (not (on ?b ?b) ) (on desk ?b) (and (on ?b bottom_shelf ?b) (object_orientation ?b ?b) ) (agent_holds ?b ?b) (agent_holds ?b) (and (and (agent_holds ?b) (and (not (agent_holds ?b) ) (< 1 1) (agent_holds ?b agent) (not (not (not (in desk ?b) ) ) ) ) ) (agent_holds ?b) ) (in_motion ?b) ) (rug_color_under ?b) )
            (once (not (adjacent ?b) ) )
            (once (agent_holds ?b ?b) )
          )
        )
      )
    )
    (preference preference4
      (then
        (once (not (and (and (is_setup_object desk ?xxx) (< (distance desk) 1) ) (in_motion ?xxx ?xxx ?xxx) ) ) )
        (once (in color bed ?xxx) )
        (any)
      )
    )
  )
)
(:terminal
  (>= (count-measure preference1:yellow) (count preference3:pyramid_block) )
)
(:scoring
  5
)
)


(define (game game-id-142) (:domain few-objects-room-v1)
(:setup
  (or
    (game-optional
      (in bed ?xxx)
    )
    (exists (?f - color)
      (game-optional
        (and
          (in_motion floor)
          (< (distance 6 ?f 1) 1)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - color)
      (and
        (preference preference1
          (exists (?p - (either beachball blue_cube_block dodgeball))
            (then
              (hold-while (agent_holds ?f) (not (< 8 (distance ?p ?p)) ) )
              (hold (or (agent_holds ?f ?f) (and (on ?f ?p) (in door ?p) ) ) )
            )
          )
        )
        (preference preference2
          (then
            (hold (not (in_motion ?f) ) )
            (once (adjacent ?f) )
            (once (= (distance ?f 3) (distance ?f agent)) )
          )
        )
        (preference preference3
          (then
            (once (in_motion ?f ?f) )
            (once (on ?f ?f) )
            (once-measure (forall (?j - shelf ?t - hexagonal_bin ?u - curved_wooden_ramp) (in_motion ?f ?u) ) (distance ?f ?f) )
            (once (is_setup_object ?f) )
          )
        )
      )
    )
    (preference preference4
      (exists (?p - tall_cylindrical_block)
        (then
          (once (and (on ?p) (< (distance ) (distance 6 ?p)) ) )
          (once (on ?p) )
          (once (and (on floor) (on agent) ) )
        )
      )
    )
  )
)
(:terminal
  (and
    (<= (- 7 )
      (total-score)
    )
    (>= (> (count-once-per-objects preference2:dodgeball:yellow) (count-once-per-objects preference1:book) )
      (count preference1:dodgeball:blue)
    )
  )
)
(:scoring
  8
)
)


(define (game game-id-143) (:domain many-objects-room-v1)
(:setup
  (forall (?w - book ?k - building)
    (or
      (forall (?x - building ?u - ball ?h - golfball)
        (forall (?u - dodgeball ?l - pillow)
          (forall (?x - rug)
            (exists (?z - building ?j - hexagonal_bin)
              (exists (?v - hexagonal_bin ?y - dodgeball ?s - doggie_bed ?s - cube_block ?y - hexagonal_bin)
                (game-optional
                  (not
                    (equal_z_position ?x)
                  )
                )
              )
            )
          )
        )
      )
      (and
        (forall (?w - (either laptop))
          (game-optional
            (not
              (and
                (not
                  (agent_holds door rug)
                )
                (and
                  (and
                    (not
                      (and
                        (and
                          (in ?k)
                          (on ?w ?k)
                        )
                        (exists (?q - hexagonal_bin)
                          (not
                            (agent_holds ?k ?k)
                          )
                        )
                      )
                    )
                    (open ?w ?k)
                  )
                  (in_motion west_wall)
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
        (once (in_motion ?xxx) )
        (hold-while (< (distance ?xxx ?xxx) 1) (and (not (in_motion ?xxx green_golfball) ) (and (and (on floor ?xxx) (not (same_color ?xxx) ) ) (agent_holds ?xxx) ) (rug_color_under ?xxx) ) (agent_holds ?xxx) (in_motion ?xxx ?xxx) )
        (hold (game_over ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 15 (+ (* (* (total-score) (count-once-per-objects preference1:basketball) )
        3
      )
      (count-overlapping preference1:orange)
    )
  )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-144) (:domain few-objects-room-v1)
(:setup
  (exists (?n - beachball ?d - hexagonal_bin)
    (exists (?o ?n - (either side_table cube_block laptop) ?y - dodgeball)
      (forall (?v - hexagonal_bin ?e - hexagonal_bin ?p - (either laptop cd yellow_cube_block))
        (exists (?t - (either cellphone cellphone))
          (game-optional
            (touch ?p)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - cube_block)
        (then
          (once (in_motion ?i) )
          (hold (in_motion agent) )
          (once (in_motion agent) )
        )
      )
    )
    (preference preference2
      (exists (?j - dodgeball)
        (then
          (once (in_motion ?j) )
          (once (and (not (agent_holds ?j ?j) ) (agent_holds ?j) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (count-once-per-objects preference1:tall_cylindrical_block) )
)
(:scoring
  6
)
)


(define (game game-id-145) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (and
      (adjacent ?xxx top_drawer)
    )
  )
)
(:constraints
  (and
    (forall (?i - pillow ?e - dodgeball)
      (and
        (preference preference1
          (then
            (hold (in_motion rug ?e) )
            (hold (not (on ?e ?e) ) )
            (once (and (between ?e) (in_motion ?e) ) )
          )
        )
        (preference preference2
          (exists (?x - game_object)
            (then
              (hold-while (in_motion ?e) (in_motion rug) (and (in_motion ?e) (= 1 1) ) (agent_holds ?x) )
              (once (in_motion floor ?x) )
              (hold-while (on bed ?e) (and (in_motion ?e rug) (on ?x ?x) ) (on ?e ?x) )
              (once (agent_holds ?x) )
              (hold (not (not (not (same_object ?e) ) ) ) )
            )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (not
          (and
            (agent_holds ?xxx agent)
            (and
              (agent_holds agent)
              (on ?xxx ?xxx)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference2:blue_dodgeball) (* (external-forall-maximize 4 ) 20 )
  )
)
(:scoring
  (+ 5 (count preference3:yellow) )
)
)


(define (game game-id-146) (:domain few-objects-room-v1)
(:setup
  (forall (?u - doggie_bed ?d - chair)
    (game-conserved
      (on ?d floor)
    )
  )
)
(:constraints
  (and
    (forall (?t ?s - dodgeball)
      (and
        (preference preference1
          (then
            (hold (adjacent ?t) )
            (hold (adjacent ?t ?s) )
            (once (and (agent_holds ?t ?t) (= (distance room_center ?t) 1) ) )
          )
        )
        (preference preference2
          (exists (?d - game_object)
            (exists (?a - building)
              (then
                (once (on ?s) )
                (once (agent_holds ?s) )
                (once (forall (?z - hexagonal_bin ?g - (either dodgeball dodgeball)) (in_motion ?g ?d) ) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= 5 3 )
)
(:scoring
  (external-forall-maximize
    (count preference1:hexagonal_bin)
  )
)
)


(define (game game-id-147) (:domain medium-objects-room-v1)
(:setup
  (exists (?f - (either alarm_clock cd))
    (and
      (game-conserved
        (and
          (in_motion ?f)
          (touch ?f bed)
        )
      )
      (and
        (and
          (exists (?a - (either key_chain desktop) ?l - ball)
            (forall (?v - (either alarm_clock cd laptop))
              (and
                (exists (?s - hexagonal_bin)
                  (game-optional
                    (agent_holds ?l ?l)
                  )
                )
                (game-optional
                  (agent_holds ?l agent)
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
    (forall (?k - chair)
      (and
        (preference preference1
          (at-end
            (in_motion ?k)
          )
        )
      )
    )
  )
)
(:terminal
  (<= 3 (- (- (+ (count preference1:alarm_clock) (>= (* (count preference1:dodgeball) (count-once-per-objects preference1:pink) )
            (+ (+ (count-once-per-objects preference1:dodgeball:pyramid_block) (* (or (+ 10 30 )
                    4
                  )
                  (+ (* (count preference1:pink) (+ (count preference1:beachball:dodgeball) 10 )
                    )
                    (count-once preference1:pink)
                  )
                )
                (* (* (* 1 1 (total-time) )
                    10
                  )
                  2
                )
              )
              (external-forall-maximize
                10
              )
            )
          )
          (+ (< (count-once-per-objects preference1:red) (count-once-per-objects preference1:dodgeball) )
            2
          )
          3
          (count-once-per-objects preference1:basketball:green)
          7
        )
        3
      )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-148) (:domain few-objects-room-v1)
(:setup
  (forall (?v - building ?f - tan_cube_block)
    (game-conserved
      (touch upright ?f)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z ?k - (either cube_block pillow))
        (exists (?f ?x - chair)
          (exists (?g - (either yellow_cube_block golfball cd laptop))
            (then
              (once (not (agent_holds ?z ?g) ) )
              (once (agent_holds ?k) )
              (once (not (and (agent_crouches ?g) (exists (?y - drawer ?p - hexagonal_bin) (and (on ?p) (and (object_orientation ?f ?p) (not (agent_holds ?p) ) (agent_holds ?g green) ) ) ) (and (on ?f) (not (agent_holds ?x) ) ) (in_motion ?x) ) ) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?d - cube_block)
        (at-end
          (adjacent ?d)
        )
      )
    )
  )
)
(:terminal
  (>= (<= (count-once preference2:golfball) 2 )
    (count preference2:beachball:yellow)
  )
)
(:scoring
  10
)
)


(define (game game-id-149) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (and
      (and
        (not
          (on ?xxx ?xxx)
        )
        (adjacent ?xxx)
        (agent_holds ?xxx)
        (agent_holds ?xxx ?xxx)
        (in_motion ?xxx ?xxx)
        (and
          (and
            (and
              (< (distance_side room_center 5) (distance room_center))
              (on ?xxx)
            )
            (on ?xxx)
          )
          (< (building_size room_center ?xxx ?xxx) 1)
          (not
            (in_motion ?xxx)
          )
        )
        (not
          (exists (?z - hexagonal_bin ?c - ball)
            (in_motion ?c)
          )
        )
      )
      (on top_shelf)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - (either desktop chair pen) ?l - building ?x - blue_cube_block)
        (then
          (once (touch ?x) )
          (hold (not (on ?x ?x) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (total-score) )
)
(:scoring
  18
)
)


(define (game game-id-150) (:domain few-objects-room-v1)
(:setup
  (and
    (or
      (game-conserved
        (in_motion desk)
      )
    )
  )
)
(:constraints
  (and
    (forall (?g - dodgeball)
      (and
        (preference preference1
          (then
            (once (in_motion ?g ?g) )
            (hold-to-end (< 1 1) )
            (hold (agent_holds ?g) )
          )
        )
        (preference preference2
          (exists (?a - (either hexagonal_bin dodgeball) ?k - book)
            (then
              (hold (touch ?g ?g) )
              (hold (not (exists (?n ?d ?z - shelf ?l - block) (not (agent_holds ?k ?k) ) ) ) )
              (once (adjacent agent ?k) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?v - doggie_bed)
        (then
          (hold-while (and (agent_holds ?v ?v) (on ?v) ) (agent_holds ?v) (not (touch ?v ?v) ) )
          (once (not (not (in_motion ?v ?v) ) ) )
          (once (same_type ?v ?v) )
        )
      )
    )
    (preference preference4
      (exists (?g - cube_block)
        (exists (?n - ball)
          (then
            (hold-while (same_color ?g ?g ?g) (agent_holds ?g ?g) )
            (once (in_motion ?n ?g) )
            (once (not (in_motion ?n) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (< (count-once preference4:top_drawer) 1 )
)
(:scoring
  (* 30 (count-once-per-objects preference2:yellow) (count-once preference4:dodgeball) )
)
)


(define (game game-id-151) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (faces ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - sliding_door)
        (then
          (hold-to-end (not (and (and (on ?l ?l) (not (and (and (and (in door) (agent_holds ?l ?l door) (not (exists (?i - cube_block) (in_motion ?i) ) ) ) (and (in_motion pink) (agent_holds ?l) ) ) (agent_holds ?l ?l) (>= 1 0) ) ) (in_motion ?l ?l) ) (on agent) (in ?l) (and (on agent) (object_orientation ?l) (in_motion desk) ) ) ) )
          (once (on bed) )
          (hold-for 5 (in ?l ?l) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:doggie_bed:green) (+ (count preference1:yellow) (and (count preference1:golfball) ) )
  )
)
(:scoring
  (* (count preference1:pink) 10 )
)
)


(define (game game-id-152) (:domain few-objects-room-v1)
(:setup
  (exists (?l - flat_block)
    (exists (?s - hexagonal_bin ?g - teddy_bear ?s - drawer)
      (game-conserved
        (not
          (on ?l ?s)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - game_object)
      (and
        (preference preference1
          (then
            (once (< 7 1) )
            (once (on ?r pink) )
            (hold (and (> 2 (distance ?r ?r)) (not (agent_holds ?r ?r) ) ) )
          )
        )
        (preference preference2
          (exists (?h - cube_block ?f - ball ?x - hexagonal_bin)
            (exists (?d - hexagonal_bin ?t - ball)
              (then
                (once (adjacent_side ?t ?x) )
                (once (in ?r ?t) )
                (once-measure (and (and (and (exists (?f - hexagonal_bin) (above ?r agent) ) (and (agent_holds ?t ?r ?r) (on desk) ) ) (not (and (and (on ?x bed) (in ?r ?t) ) (= 1 (distance_side ?x room_center)) ) ) (same_color ?r ?x) (on ?r) (and (in_motion ?r bed) (not (agent_holds upright) ) ) ) (and (on ?x) (not (in_motion ?r ?x) ) ) ) (distance 4 desk) )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference2:dodgeball) (count preference1:alarm_clock:dodgeball:pink_dodgeball) )
)
(:scoring
  (- (count-overlapping preference2:pink) )
)
)


(define (game game-id-153) (:domain many-objects-room-v1)
(:setup
  (exists (?c - cube_block)
    (game-optional
      (adjacent ?c ?c)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - cube_block)
        (exists (?t - (either golfball tall_cylindrical_block) ?k - game_object)
          (exists (?l - dodgeball ?i - teddy_bear)
            (then
              (once (and (on ?w ?k) (and (= (distance ?i green_golfball) (distance ?k door ?k)) (and (not (< (distance ?i room_center) 0) ) (in_motion bed) ) (in_motion ?k) (= (distance ?w ?k) (distance agent agent)) (agent_holds ?k) ) ) )
              (hold-while (toggled_on ?w) (in_motion ?k) )
              (hold-while (game_over ?k ?i) (in_motion ?w ?k) (not (and (> (distance desk door) (distance ?w ?i ?w)) (agent_holds agent) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* (count preference1:dodgeball) (count preference1:pink:basketball) )
      (* 1 (count preference1:hexagonal_bin) )
    )
    10
  )
)
(:scoring
  (count-once-per-objects preference1:pink)
)
)


(define (game game-id-154) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (exists (?l - chair)
      (= 1 1)
    )
  )
)
(:constraints
  (and
    (forall (?c - shelf)
      (and
        (preference preference1
          (exists (?z - pyramid_block)
            (exists (?j - flat_block)
              (at-end
                (in_motion ?c)
              )
            )
          )
        )
      )
    )
    (forall (?z ?r - hexagonal_bin ?r - (either blue_cube_block pyramid_block))
      (and
        (preference preference2
          (then
            (once (in_motion floor ?r) )
            (once (on ?r) )
            (hold (in_motion agent ?r) )
            (hold (same_type ?r) )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (not (and (and (in_motion agent) (in ?xxx ?xxx) ) (and (agent_holds ?xxx ?xxx) (not (is_setup_object desk) ) ) ) ) )
        (once (agent_holds bed) )
        (hold (not (in rug) ) )
      )
    )
    (preference preference4
      (exists (?b - ball ?h - pyramid_block)
        (exists (?i - beachball ?l - game_object)
          (exists (?p - teddy_bear ?k - (either game_object dodgeball golfball beachball))
            (exists (?s - hexagonal_bin)
              (exists (?f - doggie_bed)
                (then
                  (once (adjacent ?f ?h) )
                  (hold (in ?k) )
                  (once (same_color ?k) )
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
  (= 5 (count-once-per-objects preference1:blue_dodgeball) )
)
(:scoring
  (count-once-per-objects preference4:orange:pink)
)
)


(define (game game-id-155) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (is_setup_object ?xxx)
    )
    (forall (?y - wall)
      (exists (?n - (either alarm_clock golfball) ?f - sliding_door ?j - hexagonal_bin ?v - golfball)
        (exists (?i - hexagonal_bin)
          (game-conserved
            (< (distance ?y agent) 1)
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
        (once (not (touch ?xxx) ) )
        (hold (in ?xxx ?xxx) )
        (hold-for 6 (in_motion ?xxx) )
      )
    )
    (preference preference2
      (then
        (once (not (> (distance desk room_center) 3) ) )
        (once (not (agent_holds ?xxx) ) )
        (once (not (< (distance ?xxx agent) 10) ) )
      )
    )
    (preference preference3
      (then
        (once (on agent) )
        (once (touch ?xxx ?xxx) )
        (once (and (not (agent_holds ?xxx) ) (and (adjacent ?xxx ?xxx) (on ?xxx ?xxx) (and (in_motion ?xxx desk) (game_over ?xxx ?xxx) ) ) ) )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:red:wall:blue_dodgeball) (* 15 1 )
  )
)
(:scoring
  5
)
)


(define (game game-id-156) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (touch ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (in_motion ?xxx) )
        (hold-while (on ?xxx) (on ?xxx ?xxx) (object_orientation ?xxx) (in_motion ?xxx) )
        (hold (and (and (agent_holds ?xxx ?xxx) (< 5 (distance ?xxx)) ) (not (and (and (and (not (in_motion ?xxx ?xxx) ) (in_motion agent ?xxx) ) (not (and (in_motion green_golfball ?xxx) (and (touch ?xxx) (agent_holds ?xxx) ) ) ) (on ?xxx) ) (in_motion bed) ) ) (agent_holds ?xxx ?xxx) ) )
      )
    )
    (preference preference2
      (exists (?h - (either pink desktop) ?w - cube_block)
        (then
          (once (in_motion bed) )
          (once (not (toggled_on ?w) ) )
          (hold (in_motion ?w ?w) )
        )
      )
    )
  )
)
(:terminal
  (>= 6 (count-once-per-objects preference1:yellow) )
)
(:scoring
  (count-overlapping preference2:beachball:yellow)
)
)


(define (game game-id-157) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (not
        (agent_holds ?xxx)
      )
      (not
        (agent_holds )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - (either top_drawer golfball) ?b - ball)
        (then
          (once (agent_holds ?b) )
          (once (agent_holds ?b ?b) )
          (once (not (and (not (on ?b ?b) ) (not (same_color ?b ?b) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (< (not (count-once-per-objects preference1:dodgeball) ) (= (count preference1:beachball:pink) (* (count preference1:dodgeball) (+ 3 (* 10 3 (count preference1:tan) (* 10 (+ (* (count preference1:dodgeball) 7 )
                5
                (* (+ 5 (total-time) )
                  (count preference1:yellow_pyramid_block:basketball:orange)
                  (count preference1:red)
                  (+ (+ (count-once-per-external-objects preference1:dodgeball) (total-time) (* (total-time) (- (and 4 9 (* (count-overlapping preference1:dodgeball) 6 )
                          )
                          (count preference1:pink)
                        )
                      )
                    )
                    5
                  )
                )
                (count preference1:basketball:red:pyramid_block)
                2
                (count-once-per-objects preference1:red_pyramid_block)
              )
            )
            (* (count-once-per-external-objects preference1:pyramid_block) (- 15 )
            )
          )
          (- (count-once-per-external-objects preference1:dodgeball:dodgeball) )
          (count preference1:pink:beachball)
        )
      )
    )
  )
)
(:scoring
  40
)
)


(define (game game-id-158) (:domain many-objects-room-v1)
(:setup
  (forall (?a - hexagonal_bin ?m - building)
    (exists (?k - dodgeball ?j - (either golfball pencil wall) ?a - hexagonal_bin)
      (game-optional
        (not
          (in ?a ?m)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (on ?xxx) )
        (once (same_type ?xxx ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (total-time) )
)
(:scoring
  (count preference1:basketball)
)
)


(define (game game-id-159) (:domain few-objects-room-v1)
(:setup
  (forall (?h - (either yellow_cube_block pyramid_block) ?h - (either laptop alarm_clock) ?n ?r ?a ?m ?x ?s ?c ?y ?g ?w - dodgeball)
    (exists (?p - wall)
      (exists (?q - building)
        (game-conserved
          (in_motion ?x)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (touch ?xxx ?xxx) )
        (once (on ?xxx ?xxx) )
        (once (forall (?x - (either cylindrical_block triangle_block) ?e - dodgeball ?o - hexagonal_bin) (exists (?s - pillow) (= 1 (distance ?o ?s)) ) ) )
      )
    )
  )
)
(:terminal
  (= (count preference1:doggie_bed) (count preference1:orange) )
)
(:scoring
  (* (count-once-per-objects preference1:pyramid_block) (* 5 1 (count-once-per-objects preference1:basketball:green) )
  )
)
)


(define (game game-id-160) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in_motion rug ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m ?p - color ?x - block ?r - doggie_bed)
        (then
          (once (in_motion rug) )
          (hold-while (exists (?z - block) (between ?r) ) (on ?r ?r) )
          (hold (not (exists (?b - block ?q - building) (not (and (touch ?r) (adjacent ?r) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (> (external-forall-maximize (* 15 (count-longest preference1) (count preference1:green) )
    )
    (and
      (count-once-per-objects preference1:purple)
      (* (* 10 (count-once-per-objects preference1:dodgeball) )
        25
      )
    )
  )
)
(:scoring
  7
)
)


(define (game game-id-161) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (agent_holds ?xxx ?xxx)
      )
    )
    (forall (?c - red_dodgeball)
      (and
        (preference preference2
          (at-end
            (agent_holds ?c ?c)
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
        (>= 5 (count-once-per-objects preference1:beachball:cube_block:dodgeball) )
        (>= 300 (count preference1:hexagonal_bin) )
      )
      (or
        (>= (not (count preference1:yellow) ) (* (* 1 4 5 )
            (* (count preference2:tall_cylindrical_block) (count preference2:dodgeball:green) (count preference2:dodgeball:blue_dodgeball) 3 2 (+ (count-once-per-objects preference1:yellow_cube_block) 4 (count preference2:dodgeball:cylindrical_block:basketball) (* (count preference1:basketball) 2 5 )
                (count-once-per-objects preference2:basketball)
                (+ (count-once-per-objects preference2:golfball) (- 180 )
                )
              )
            )
          )
        )
        (>= (count-once-per-objects preference1:doggie_bed) 3 )
      )
      (= (count-once-per-external-objects preference1:purple:blue_dodgeball) (count preference1:dodgeball:book) )
    )
    (>= (count-once-per-external-objects preference1:pink) (* (* (+ 6 3 )
          15
        )
        3
      )
    )
    (or
      (not
        (or
          (> 12 (* (count-once-per-objects preference1:yellow) (count-once-per-external-objects preference2:dodgeball) (count preference2:golfball) )
          )
          (> 5 (count-once-per-objects preference2:beachball) )
        )
      )
      (>= 15 (count preference1:dodgeball:golfball) )
      (not
        (<= (+ (* (count-once-per-external-objects preference1:dodgeball) (* (count-once-per-objects preference1:book:side_table:bed) (count preference2:basketball:yellow) )
              3
            )
            (not
              (* 5 5 )
            )
          )
          2
        )
      )
    )
  )
)
(:scoring
  (total-score)
)
)


(define (game game-id-162) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (in ?xxx ?xxx) ) )
        (once (in_motion rug) )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 5 (+ (+ (* 2 (count-longest preference1:wall:book) )
        (* (* (count preference1:orange) 3 )
          100
        )
      )
      (* 5 (+ (* (total-time) (or (* (>= (or (count preference1:pink:dodgeball) ) (count-longest preference1:green) )
                (external-forall-maximize
                  (+ (* 5 3 (count-total preference1:blue_dodgeball) )
                    (count preference1:dodgeball)
                  )
                )
                (total-score)
                15
                (external-forall-maximize
                  (count-once-per-objects preference1:green)
                )
              )
              (* (* (count preference1:dodgeball) (external-forall-maximize (count preference1:red) ) (count preference1:white) 5 (count preference1:dodgeball) (count preference1:red) (not (* 30 10 )
                  )
                  (* (+ (+ (count-once-per-objects preference1:basketball) (* 9 3 )
                      )
                      (+ (count-once-per-objects preference1:red_pyramid_block) 4 (* (count-once-per-objects preference1) 1 )
                      )
                      (count-same-positions preference1:dodgeball:basketball)
                      (count-once-per-objects preference1:blue_dodgeball)
                    )
                    (count-once-per-objects preference1:golfball)
                  )
                  (count-measure preference1:alarm_clock)
                  (- (count preference1:basketball) )
                  2
                  180
                )
                (count preference1:pink)
              )
              (count-once-per-objects preference1:dodgeball)
            )
            2
            6
            (count preference1:hexagonal_bin)
            (= 50 )
            1
            10
            (-
              4
              8
            )
          )
          1
        )
      )
      (count-longest preference1)
    )
  )
)
(:scoring
  (total-score)
)
)


(define (game game-id-163) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx upright)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (and (and (and (and (not (in_motion agent pink_dodgeball) ) (not (in_motion ?xxx ?xxx) ) ) (in_motion ?xxx ?xxx) (in_motion ?xxx ?xxx) ) (and (adjacent ?xxx ?xxx) (touch ?xxx) ) ) (on ?xxx ?xxx) ) )
      )
    )
    (preference preference2
      (exists (?s - (either alarm_clock key_chain) ?o - blue_cube_block ?o - curved_wooden_ramp ?j - doggie_bed)
        (then
          (hold (and (> 1 1) (and (agent_holds ?j) (in_motion ?j) ) (and (adjacent north_wall) (and (agent_holds ?j) (in_motion ?j) ) ) ) )
          (once (and (agent_holds ?j) (in_motion ?j ?j) (agent_holds ?j ?j) ) )
          (hold-while (not (on ?j ?j) ) (>= (distance agent ?j) (distance ?j desk)) (in_motion upright) )
        )
      )
    )
    (preference preference3
      (at-end
        (>= (distance desk room_center ?xxx) (distance ?xxx))
      )
    )
  )
)
(:terminal
  (or
    (or
      (and
        (<= 10 (* (* 15 10 )
          )
        )
        (>= (total-time) 5 )
        (>= (count-once-per-objects preference3:red) (count-once-per-external-objects preference1:side_table) )
      )
      (not
        (or
          (> (+ 10 (* (count-once-per-objects preference2:pink_dodgeball) (count preference1:basketball) )
              1
            )
            (count-once preference2:green)
          )
          (and
            (>= (count preference3:pink_dodgeball) (- (* (count preference2:purple:cylindrical_block) (count-once preference1:blue_cube_block) )
              )
            )
            (>= (* (count-total preference3:blue_dodgeball) (+ 15 (not (count preference2:red:yellow) ) )
              )
              4
            )
            (>= (total-score) (count-once preference1:basketball) )
          )
        )
      )
      (>= (+ (* (or (count-once-per-objects preference1:pink_dodgeball:blue_dodgeball:beachball) (* (count preference3:cylindrical_block) 15 )
            )
            (* (count preference3:dodgeball) (- (* 10 100 )
              )
              9
            )
          )
          5
        )
        (+ (count-unique-positions preference2) 5 )
      )
    )
    (>= (external-forall-maximize (* (count preference2:pyramid_block) 10 )
      )
      (* 1 3 (* 7 5 )
        (* (count preference1:beachball:blue_dodgeball) (>= (count-once-per-objects preference1:beachball) (* (count-same-positions preference1:dodgeball:basketball) 5 5 )
          )
        )
        (count-once-per-objects preference1:beachball)
        (external-forall-maximize
          (count-overlapping preference3:basketball:doggie_bed)
        )
      )
    )
  )
)
(:scoring
  (external-forall-maximize
    (* (count-total preference2:book) 7 )
  )
)
)


(define (game game-id-164) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (same_object ?xxx)
        (and
          (not
            (agent_holds agent)
          )
          (or
            (exists (?z - block ?l - ball ?l - cylindrical_block)
              (agent_holds ?l)
            )
            (in_motion ?xxx ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - doggie_bed)
        (then
          (hold (in_motion ?a) )
          (once (and (in_motion blue ?a ?a) (not (not (or (on ?a) ) ) ) ) )
          (hold-while (< 9 (distance door ?a)) (not (on ?a) ) )
        )
      )
    )
  )
)
(:terminal
  (not
    (> (+ (* (+ 8 10 )
          (or
            6
          )
        )
        (count-unique-positions preference1:rug:dodgeball)
      )
      (* 2 (* 50 5 (count-once-per-external-objects preference1:blue_dodgeball) )
      )
    )
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id-165) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (= 1 0)
  )
)
(:constraints
  (and
    (forall (?m - hexagonal_bin ?l - tall_cylindrical_block)
      (and
        (preference preference1
          (exists (?f - tall_cylindrical_block ?w - curved_wooden_ramp ?a - (either cellphone dodgeball))
            (exists (?j - hexagonal_bin)
              (then
                (once (not (in_motion desk) ) )
                (once (in ?j ?j) )
                (once (and (not (and (and (in_motion ?l ?l) (and (agent_holds ?j ?l) (in ?l ?l) ) (and (adjacent ?l ?l) (and (and (on ?j agent) (touch agent ?a) ) (agent_holds desk) ) (in_motion rug) (and (not (not (on ?l) ) ) (not (in ?j) ) ) ) ) (in_motion ?a) ) ) (and (adjacent agent) (< 3 (distance desk ?a)) (agent_holds ?a) ) ) )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (hold (not (adjacent ?xxx ?xxx) ) )
        (once (not (exists (?h ?q - cube_block) (and (and (exists (?g - dodgeball ?w - hexagonal_bin) (same_color agent ?h) ) (and (on ?h ?h) (not (not (exists (?i - teddy_bear ?m ?k ?v - block ?v - dodgeball) (not (< (distance desk agent) 2) ) ) ) ) (not (not (not (or (on desk ?h) (and (not (not (and (in_motion ?q ?q) (not (exists (?u - block ?r - building) (= 10 (distance ?h room_center) (distance ?r bed)) ) ) ) ) ) (and (on ?h ?h) (and (in_motion ?h ?h) (rug_color_under ?h) ) (in_motion ?h) (agent_holds desk) (in top_shelf) (not (< (distance 9 bed) 1) ) (agent_holds rug green_golfball ?q) (and (> (distance ?h front_left_corner) 1) (agent_holds ) (agent_holds ?q ?h) ) ) ) ) ) ) ) (in_motion ?q) ) (not (and (agent_holds ?h) (agent_holds ?q) ) ) ) (not (open ?h) ) ) ) ) )
        (hold (not (not (adjacent ?xxx ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (> (count-same-positions preference2:pink) (* (* (+ 8 (total-score) )
        (* (count-once-per-objects preference2:dodgeball:dodgeball) (+ 60 1 (* (count-once-per-external-objects preference1:red_pyramid_block) (count preference1:doggie_bed:beachball:basketball) )
          )
        )
      )
      50
      (>= (not (* (count preference2:blue_cube_block) (count-once-per-objects preference1:dodgeball:blue_pyramid_block) )
        )
        (count preference2:dodgeball)
      )
    )
  )
)
(:scoring
  (* (count-once-per-objects preference2:purple) 4 )
)
)


(define (game game-id-166) (:domain few-objects-room-v1)
(:setup
  (or
    (exists (?k - hexagonal_bin)
      (and
        (and
          (and
            (game-conserved
              (agent_holds ?k)
            )
          )
        )
      )
    )
    (exists (?f - red_pyramid_block)
      (forall (?t - dodgeball)
        (exists (?q - hexagonal_bin ?w - hexagonal_bin)
          (game-conserved
            (adjacent ?f)
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
        (once (touch ?xxx) )
        (hold (in_motion ?xxx) )
        (once (in ?xxx) )
        (hold (< (distance ?xxx 8) 1) )
        (hold (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (not
    (>= 3 (count-once-per-objects preference1:dodgeball) )
  )
)
(:scoring
  (* 1 (count-measure preference1:beachball) )
)
)


(define (game game-id-167) (:domain medium-objects-room-v1)
(:setup
  (exists (?x - cube_block ?o - building)
    (forall (?j - blue_pyramid_block ?q - (either hexagonal_bin chair cylindrical_block cube_block) ?g ?m - game_object ?n ?p - ball)
      (game-conserved
        (and
          (in_motion ?p)
          (on door agent)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (and (not (in_motion ?xxx) ) (touch ?xxx agent) (and (and (agent_holds upright ?xxx) (in_motion ?xxx) (in ?xxx ?xxx) (and (and (adjacent ?xxx ?xxx) (not (in_motion ?xxx) ) (in_motion ?xxx) ) (agent_holds ?xxx) ) (agent_holds ?xxx ?xxx) (not (in_motion side_table ?xxx) ) (touch ?xxx) ) (not (agent_holds ?xxx ?xxx) ) (adjacent_side ?xxx ?xxx) ) ) )
        (once (in_motion ?xxx ?xxx) )
        (once (on ?xxx) )
      )
    )
    (forall (?w - tall_cylindrical_block ?n - flat_block)
      (and
        (preference preference2
          (exists (?q - dodgeball ?a - game_object ?k - game_object)
            (at-end
              (not
                (in_motion rug)
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?a - cube_block ?u - (either bridge_block cylindrical_block) ?s - dodgeball)
        (exists (?g - hexagonal_bin)
          (exists (?m - book ?f - doggie_bed)
            (then
              (once (and (in_motion agent bed) (not (not (on ?g ?s) ) ) ) )
              (once (in_motion ?f ?f) )
              (once (in_motion bed) )
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
      (>= 100 (count preference1:pink_dodgeball) )
    )
    (>= (* (count preference3:brown:golfball) (count preference1:hexagonal_bin) )
      (count preference1:triangle_block)
    )
  )
)
(:scoring
  (+ (* (external-forall-minimize (* 16 3 )
      )
      (+ (count preference1:pink) (* 10 (external-forall-maximize (- 3 )
          )
        )
      )
    )
    10
  )
)
)


(define (game game-id-168) (:domain many-objects-room-v1)
(:setup
  (exists (?c - shelf)
    (game-optional
      (agent_holds ?c)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - ball)
        (exists (?q - ball ?a - ball)
          (exists (?e ?i - cube_block)
            (forall (?g - block)
              (at-end
                (and
                  (in_motion ?a ?i)
                  (or
                    (and
                      (on desk)
                      (not
                        (and
                          (in west_wall ?a)
                          (touch ?a ?g)
                        )
                      )
                      (in_motion ?a)
                      (in_motion left agent)
                      (not
                        (and
                          (and
                            (touch ?e)
                            (or
                              (not
                                (in_motion ?g)
                              )
                            )
                          )
                          (and
                            (>= 1 2)
                            (on agent)
                          )
                        )
                      )
                      (not
                        (agent_holds agent ?i)
                      )
                      (not
                        (and
                          (agent_holds ?e ?i)
                          (and
                            (in top_shelf)
                            (agent_holds ?p ?e)
                          )
                        )
                      )
                      (agent_holds agent)
                    )
                    (agent_holds ?e ?p)
                  )
                  (and
                    (<= 8 (distance ?p 4))
                    (and
                      (not
                        (and
                          (exists (?u ?b - ball)
                            (and
                              (not
                                (on ?g ?i)
                              )
                              (not
                                (and
                                  (agent_holds ?a)
                                  (exists (?t - hexagonal_bin)
                                    (in_motion ?p ?b)
                                  )
                                )
                              )
                            )
                          )
                          (on ?p ?a)
                        )
                      )
                      (or
                        (not
                          (between ?i agent)
                        )
                        (on ?e ?g)
                      )
                    )
                    (not
                      (agent_holds ?a bridge_block)
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
  (>= (+ (* (count-once-per-objects preference1:doggie_bed) (count preference1:block:pink_dodgeball) )
      2
    )
    (count-same-positions preference1:red)
  )
)
(:scoring
  (not
    (count preference1:red:cube_block)
  )
)
)


(define (game game-id-169) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (not
        (agent_holds ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - (either pencil triangular_ramp))
        (then
          (once (agent_holds rug) )
          (once (not (forall (?e - hexagonal_bin) (not (agent_holds ?j ?j) ) ) ) )
          (once (and (in ?j agent) (and (and (agent_holds ?j ?j) (not (and (in ?j ?j ?j) (agent_holds ?j bed) ) ) ) (agent_holds ?j agent) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 4 2 )
)
(:scoring
  (count preference1:dodgeball)
)
)


(define (game game-id-170) (:domain few-objects-room-v1)
(:setup
  (forall (?i - dodgeball ?c - pillow)
    (and
      (and
        (game-conserved
          (not
            (same_type ?c)
          )
        )
      )
      (game-conserved
        (in_motion ?c)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - hexagonal_bin)
        (then
          (hold (forall (?b - golfball) (in ?o) ) )
          (hold-to-end (adjacent ?o) )
          (once (not (in_motion ?o) ) )
        )
      )
    )
    (preference preference2
      (then
        (once (agent_holds bed) )
        (hold (on ?xxx rug) )
        (hold (not (not (agent_holds ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (>= (count preference1:hexagonal_bin) (or (external-forall-maximize (* (count preference2:yellow_cube_block) 5 (count preference2:pink_dodgeball) 2 (count preference1:pink_dodgeball) (count preference2:dodgeball) )
      )
      (count-measure preference1)
    )
  )
)
(:scoring
  (+ (count preference2:basketball:basketball) (count-once-per-objects preference2:beachball) )
)
)


(define (game game-id-171) (:domain many-objects-room-v1)
(:setup
  (forall (?c - drawer)
    (game-optional
      (agent_holds ?c)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (adjacent ?xxx agent) (on ?xxx) (on ?xxx) (not (touch floor) ) (agent_holds floor) (and (and (touch ?xxx ?xxx) (not (on ?xxx) ) ) (not (in_motion ?xxx agent) ) ) (not (and (= (distance desk ?xxx) 4) (and (agent_holds ?xxx) (not (adjacent ?xxx ?xxx) ) ) ) ) (in_motion ?xxx) ) )
        (hold (and (not (adjacent ?xxx) ) (not (not (and (touch ?xxx) (not (agent_holds ?xxx) ) ) ) ) ) )
        (hold-while (not (on ?xxx) ) (adjacent rug) (in_motion ?xxx upright) )
      )
    )
  )
)
(:terminal
  (< (count preference1:dodgeball) (count-once-per-objects preference1:yellow) )
)
(:scoring
  2
)
)


(define (game game-id-172) (:domain many-objects-room-v1)
(:setup
  (exists (?d - shelf ?g - block)
    (or
      (game-conserved
        (not
          (not
            (game_over ?g)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - curved_wooden_ramp)
      (and
        (preference preference1
          (then
            (once (in_motion ?i ?i) )
            (once (not (in_motion ?i ?i) ) )
            (once (agent_holds ?i ?i) )
          )
        )
        (preference preference2
          (exists (?a - dodgeball)
            (then
              (once (adjacent_side ?a ?i) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 30 3 )
)
(:scoring
  (count preference2:dodgeball:side_table)
)
)


(define (game game-id-173) (:domain many-objects-room-v1)
(:setup
  (exists (?w - hexagonal_bin)
    (exists (?h - game_object ?d - doggie_bed)
      (forall (?o - cube_block)
        (exists (?t - chair ?z - dodgeball)
          (game-conserved
            (< (distance green_golfball room_center side_table) (distance ?o front_left_corner))
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
        (agent_holds ?xxx)
      )
    )
  )
)
(:terminal
  (= 25 (* 1 (or 2 (count preference1:blue_dodgeball:yellow) (+ 3 (* (count-once-per-objects preference1:dodgeball) (count-once preference1) (count preference1:green:blue_pyramid_block) )
        )
      )
    )
  )
)
(:scoring
  (count-once-per-objects preference1:beachball)
)
)


(define (game game-id-174) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (above ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m - cube_block)
        (then
          (once (and (not (> (distance ?m green_golfball) (distance ?m agent)) ) (agent_holds ?m ?m) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (< 3 (count-once-per-objects preference1:rug) )
    (or
      (or
        (>= (total-time) (* 2 (total-score) )
        )
        (> 5 (count-overlapping preference1:golfball) )
        (>= (count-once-per-objects preference1:beachball:golfball) (+ 40 2 )
        )
      )
      (>= (+ 10 (* 6 (count preference1:blue_dodgeball) )
        )
        0.5
      )
    )
    (or
      (>= 7 3 )
      (>= (count preference1:cube_block) (* (* (count preference1:tan) (count-measure preference1:beachball) (count preference1:golfball) )
          (count preference1:block)
        )
      )
      (>= 8 6 )
    )
  )
)
(:scoring
  (* 15 8 )
)
)


(define (game game-id-175) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx bed)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (in_motion ?xxx floor) (adjacent ?xxx ?xxx) ) )
        (hold (agent_holds ?xxx) )
        (once (not (and (and (< (distance room_center side_table) (distance ?xxx room_center)) (in_motion ?xxx) ) (and (and (agent_holds ?xxx) (not (adjacent ?xxx top_drawer) ) (same_color ?xxx ?xxx) ) (< 1 1) (in_motion ?xxx ?xxx) ) (agent_holds agent) ) ) )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:golfball) 12 )
)
(:scoring
  (count-measure preference1:doggie_bed)
)
)


(define (game game-id-176) (:domain many-objects-room-v1)
(:setup
  (forall (?i - dodgeball)
    (exists (?b - doggie_bed ?r - dodgeball)
      (and
        (and
          (and
            (exists (?u ?n ?k ?c - hexagonal_bin)
              (and
                (and
                  (or
                    (or
                      (not
                        (game-conserved
                          (touch ?i)
                        )
                      )
                      (game-optional
                        (agent_holds bed)
                      )
                      (game-optional
                        (adjacent_side ?u ?k)
                      )
                    )
                    (not
                      (forall (?w - ball)
                        (and
                          (game-conserved
                            (in ?k)
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
            (and
              (in ?r)
              (on ?r)
            )
          )
        )
        (and
          (game-conserved
            (on ?r)
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
        (once (agent_holds ?xxx) )
        (once (and (not (and (in ?xxx ?xxx) (agent_holds agent ?xxx) ) ) (agent_holds blue) (and (in_motion ?xxx ?xxx) (adjacent ?xxx) (and (and (agent_holds ?xxx) (on rug ?xxx) ) (and (faces ?xxx front rug) (not (on floor) ) ) ) (in ?xxx) (on ?xxx ?xxx) (and (in desk) (forall (?x - hexagonal_bin) (in_motion ?x) ) ) (not (or (and (in_motion ?xxx) (agent_holds agent) (adjacent ?xxx rug) ) (and (agent_holds tan) (on ?xxx ?xxx) ) (not (agent_holds green) ) ) ) ) ) )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (>= 50 (* (count preference1:basketball:rug) (+ (* (* (* (* (+ (count-once preference1:pink:triangle_block:dodgeball) (* (count preference1:basketball) (count-once-per-objects preference1:beachball:yellow) )
                    (or
                      (* (* (count-once preference1:book) 10 )
                        (count preference1:pink)
                      )
                      (external-forall-maximize
                        (count-once-per-external-objects preference1:dodgeball)
                      )
                    )
                  )
                  (- (count preference1:pink_dodgeball:block) )
                  (count-longest preference1:red_pyramid_block)
                  1
                  (* (count preference1:dodgeball:brown) (external-forall-maximize (= (count preference1:golfball) (count preference1:blue_cube_block:dodgeball) )
                    )
                    30
                  )
                  (external-forall-minimize
                    (count preference1:yellow)
                  )
                  (count preference1)
                  (count preference1:book)
                  4
                )
                5
              )
              1
            )
            6
          )
        )
      )
    )
    (or
      (>= (- (- (count-once-per-objects preference1:golfball:pink) )
        )
        (* 10 5 (* (and (count-once-per-objects preference1:pink_dodgeball) ) (count preference1:tan) )
        )
      )
      (and
        (>= 50 (count-shortest preference1:blue_dodgeball) )
      )
    )
  )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-177) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (touch ?xxx upright) )
        (once (touch ?xxx pink_dodgeball) )
        (once (< (distance ?xxx green_golfball) (distance ?xxx 0 ?xxx)) )
      )
    )
  )
)
(:terminal
  (> (count-once-per-objects preference1:blue_pyramid_block) (> (count-once-per-external-objects preference1:golfball) 3 )
  )
)
(:scoring
  1
)
)


(define (game game-id-178) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (< 2 (distance ?xxx 3))
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - game_object ?b - (either dodgeball bridge_block) ?v - cube_block)
        (then
          (once (not (not (in ?v) ) ) )
          (hold-while (agent_holds ?v) (and (same_color ?v ?v) (not (< (distance ?v ?v) (distance_side )) ) (not (not (equal_z_position ?v) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (+ 2 (* (* (+ (* (count preference1:basketball) (external-forall-maximize (- (* 3 (external-forall-minimize (count preference1:red) ) (- (* (count-overlapping preference1:yellow) (total-score) )
                    )
                    5
                  )
                )
              )
            )
            (total-score)
            (= 10 (count preference1:blue) )
            (count preference1:red:blue_dodgeball:alarm_clock)
            3
            10
          )
          (= (count-once-per-objects preference1:yellow) (count-once-per-objects preference1:beachball) )
        )
        (count-overlapping preference1:doggie_bed)
      )
    )
    (* 10 (total-time) (* 3 5 )
      (count-once preference1:book)
      (* (* (count-once preference1:beachball:cube_block) (count-once-per-objects preference1:cylindrical_block) )
        50
      )
      (count preference1:cylindrical_block:pink)
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-179) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (exists (?a ?x - block ?s - (either cube_block wall))
          (in )
        )
        (not
          (on agent)
        )
      )
    )
    (game-optional
      (touch ?xxx floor)
    )
  )
)
(:constraints
  (and
    (forall (?k - hexagonal_bin)
      (and
        (preference preference1
          (at-end
            (in ?k ?k)
          )
        )
        (preference preference2
          (exists (?r - sliding_door)
            (then
              (once (in ?r ?k) )
              (hold (agent_holds ?r ?r) )
              (once (in ?k ?r) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (count-once-per-objects preference1) )
)
(:scoring
  (= (count preference1:basketball) 15 )
)
)


(define (game game-id-180) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (and
        (exists (?p - watch)
          (agent_holds ?p)
        )
        (and
          (and
            (and
              (not
                (and
                  (in_motion drawer ?xxx)
                  (on ?xxx)
                )
              )
              (in_motion ?xxx ?xxx)
              (in_motion ?xxx)
            )
            (adjacent ?xxx)
          )
          (agent_holds ?xxx ?xxx)
          (agent_holds ?xxx)
        )
        (not
          (not
            (agent_holds ?xxx)
          )
        )
        (not
          (or
            (in_motion ?xxx)
            (adjacent rug ?xxx)
            (open rug)
          )
        )
        (and
          (not
            (agent_holds ?xxx ?xxx)
          )
          (in_motion ?xxx ?xxx)
        )
        (and
          (and
            (agent_holds ?xxx)
            (not
              (agent_holds ?xxx ?xxx)
            )
          )
          (not
            (agent_holds ?xxx ?xxx)
          )
          (<= (distance ?xxx ?xxx) 10)
        )
        (agent_holds agent)
        (and
          (and
            (same_color ?xxx)
            (and
              (>= 1 (distance ?xxx))
              (same_color ?xxx)
            )
          )
          (touch ?xxx)
        )
      )
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - block)
        (exists (?m - dodgeball)
          (then
            (hold (agent_holds ?m) )
            (hold (< 10 1) )
            (once (not (not (and (and (on ?m rug) (in_motion ?u) (on ?u) ) (object_orientation agent) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (< 20 (count preference1:blue_dodgeball) )
    (or
      (not
        (or
          (> (* (count preference1:hexagonal_bin:doggie_bed:yellow) (* (> (count preference1:pink) 100 )
                1
              )
            )
            2
          )
          (or
            (> (* 10 4 )
              (count-unique-positions preference1:alarm_clock:basketball)
            )
            (>= (count-once-per-objects preference1:beachball) (* (count-once-per-external-objects preference1:pink) 6 )
            )
            (>= (total-score) (* (* (* (count-once-per-objects preference1:hexagonal_bin) (+ (count-total preference1:orange:yellow) (count preference1:tall_cylindrical_block) )
                  )
                  (count-once-per-objects preference1:dodgeball:blue_pyramid_block:orange)
                )
                (count-overlapping preference1:dodgeball)
                (count-once preference1:hexagonal_bin)
              )
            )
            (>= (count-once-per-objects preference1:red_pyramid_block) (not 3 ) )
          )
          (< 5 (* (* 180 (+ (external-forall-maximize (count preference1) ) 8 (total-score) (* (count preference1:alarm_clock) (* (count-once-per-objects preference1:green:hexagonal_bin) 2 (count-unique-positions preference1:orange) )
                  )
                )
                (+ (count preference1:pink_dodgeball) (count preference1:dodgeball) )
                (/
                  (count-once-per-objects preference1:yellow)
                  (* (- (+ 2 (+ 7 3 )
                      )
                    )
                    (count-once-per-objects preference1:purple)
                  )
                )
                (count preference1:tall_cylindrical_block:orange)
              )
              (count preference1:dodgeball)
            )
          )
        )
      )
    )
    (>= (count-once-per-objects preference1:dodgeball) 5 )
  )
)
(:scoring
  (* (count-once-per-objects preference1:basketball:pink_dodgeball) (count-once-per-objects preference1:beachball) )
)
)


(define (game game-id-181) (:domain medium-objects-room-v1)
(:setup
  (exists (?o ?b ?c ?x ?j - doggie_bed)
    (game-conserved
      (not
        (on bed)
      )
    )
  )
)
(:constraints
  (and
    (forall (?w - cube_block ?r - (either golfball cellphone mug))
      (and
        (preference preference1
          (exists (?n - hexagonal_bin ?p - block)
            (exists (?a - game_object)
              (exists (?u - hexagonal_bin)
                (exists (?i - teddy_bear ?z - cube_block)
                  (then
                    (hold (on agent ?a) )
                    (once (adjacent_side pink) )
                    (once (in_motion front) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (forall (?e - shelf)
      (and
        (preference preference2
          (then
            (hold (or (and (<= 4 8) (and (touch ?e) (agent_holds pink_dodgeball ?e) ) ) (= 1 (distance front room_center)) ) )
            (hold-while (touch ?e) (in_motion floor ?e) )
            (once (in_motion ?e ?e) )
          )
        )
      )
    )
    (forall (?h - hexagonal_bin)
      (and
        (preference preference3
          (exists (?t - doggie_bed ?a - game_object)
            (exists (?c - cube_block)
              (then
                (hold (forall (?i - shelf) (agent_holds ?h ?c) ) )
                (once (not (agent_holds blue ?a) ) )
                (hold-while (and (and (agent_holds ?c) (touch ?c) (and (adjacent_side ?a agent) (not (agent_holds ?h) ) ) ) (not (touch ?c) ) ) (and (touch ?a) (in_motion ?a) ) )
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
    (or
      (>= (count preference2:yellow:beachball) 4 )
      (>= (* (+ (+ 1 (+ (* (* (+ 18 20 (* (* (count preference2:pink) (external-forall-minimize 5 ) )
                        (* (external-forall-maximize (>= 2 (* (- (total-score) )
                                (= (count preference3:dodgeball) )
                              )
                            )
                          )
                          (count-total preference3:purple)
                        )
                      )
                    )
                    (total-score)
                  )
                  (count-same-positions preference3:dodgeball:beachball)
                )
                8
              )
              (count preference3:yellow:blue_cube_block:blue_dodgeball)
              (count-once preference2:basketball)
              10
              (total-time)
            )
            (count-once-per-objects preference1:yellow)
          )
          (* (count preference2:beachball:dodgeball) 20 )
        )
        (count-unique-positions preference2:block)
      )
    )
    (>= 8 (* (total-time) )
    )
  )
)
(:scoring
  7
)
)


(define (game game-id-182) (:domain few-objects-room-v1)
(:setup
  (forall (?s ?l - green_triangular_ramp)
    (game-conserved
      (and
        (in_motion ?l ?l)
        (in_motion ?s)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - (either hexagonal_bin blue_cube_block))
        (then
          (hold-while (in_motion ?z ?z) (not (adjacent ?z ?z) ) )
          (once (not (and (in_motion ?z) (and (in ?z) (same_object ?z ?z) ) (adjacent_side ?z) ) ) )
          (once (agent_holds ?z ?z) )
        )
      )
    )
    (preference preference2
      (then
        (hold (touch ?xxx ?xxx) )
        (once (agent_holds ?xxx) )
        (once (not (in_motion ?xxx) ) )
      )
    )
  )
)
(:terminal
  (<= (* 3 )
    2
  )
)
(:scoring
  (count preference2:hexagonal_bin)
)
)


(define (game game-id-183) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?h - hexagonal_bin)
      (and
        (preference preference1
          (exists (?z - hexagonal_bin)
            (then
              (once (not (and (in ?z) (in ?z ?z) ) ) )
              (once (in ?h) )
              (hold-while (not (< 6 (distance agent ?h)) ) (touch ?z ?h) (and (and (on ?z ?h) (in_motion top_shelf ?h) ) (exists (?x - ball) (< 3 1) ) ) (not (not (on ?z) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:wall) 2 )
)
(:scoring
  7
)
)


(define (game game-id-184) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (not
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - game_object)
      (and
        (preference preference1
          (exists (?e - chair ?s - hexagonal_bin)
            (exists (?o - shelf)
              (then
                (once (agent_holds ?s) )
                (any)
                (hold-while (agent_holds ?f) (in ?f) )
              )
            )
          )
        )
      )
    )
    (forall (?i - block ?z - curved_wooden_ramp)
      (and
        (preference preference2
          (then
            (once (agent_holds bed) )
            (hold (between ?z) )
            (once (in agent ?z) )
          )
        )
      )
    )
  )
)
(:terminal
  (< (count preference2:pink:doggie_bed) (+ (count preference1:dodgeball) (>= (count-same-positions preference1:yellow) 2 )
    )
  )
)
(:scoring
  30
)
)


(define (game game-id-185) (:domain many-objects-room-v1)
(:setup
  (not
    (exists (?i - dodgeball)
      (and
        (game-conserved
          (agent_holds ?i ?i)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (adjacent ?xxx ?xxx) )
        (hold (agent_holds ?xxx) )
        (once (and (exists (?s ?c - dodgeball) (on ?s) ) (and (in_motion ?xxx ?xxx) (agent_holds ?xxx ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (and
    (or
      (< (count preference1:yellow_cube_block) 5 )
      (>= (total-score) 2 )
    )
    (or
      (> (* (count preference1:dodgeball) 6 )
        (count preference1:basketball)
      )
      (>= (+ (not (count-once-per-objects preference1:red:doggie_bed) ) (count preference1:dodgeball) )
        (+ (or (count preference1:blue_dodgeball:blue_dodgeball) ) 2 )
      )
      (= (* 60 (count preference1:pink) )
        (count-once-per-objects preference1:pink_dodgeball)
      )
    )
  )
)
(:scoring
  (count preference1:golfball)
)
)


(define (game game-id-186) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (adjacent ?xxx) ) )
        (once (same_type ?xxx) )
        (hold (in ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preference1:tan) 5 )
)
(:scoring
  (count-once-per-external-objects preference1:dodgeball)
)
)


(define (game game-id-187) (:domain few-objects-room-v1)
(:setup
  (exists (?m - game_object ?q - cube_block)
    (game-optional
      (and
        (and
          (agent_holds ?q ?q)
          (agent_holds ?q)
          (and
            (adjacent ?q ?q)
            (and
              (and
                (rug_color_under ?q ?q)
                (agent_holds ?q)
                (not
                  (in_motion ?q)
                )
                (not
                  (in ?q ?q)
                )
              )
              (adjacent agent ?q)
              (and
                (in ?q)
                (on ?q ?q)
              )
            )
          )
        )
        (agent_holds ?q)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (and
          (on ?xxx ?xxx)
          (not
            (touch ?xxx ?xxx)
          )
        )
      )
    )
  )
)
(:terminal
  (> (* (count preference1:beachball:pink_dodgeball) (count preference1:purple) )
    (count preference1)
  )
)
(:scoring
  (* 10 (count-once-per-external-objects preference1:beachball) )
)
)


(define (game game-id-188) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (not
        (and
          (in ?xxx)
          (and
            (agent_holds ?xxx)
            (and
              (in_motion ?xxx ?xxx)
              (not
                (exists (?e ?j - doggie_bed ?s - wall)
                  (agent_holds rug)
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
        (once (on ?xxx) )
        (once (in_motion ?xxx ?xxx) )
        (once (and (and (in_motion ?xxx) (and (> (distance 10 desk) 1) (in_motion ?xxx ?xxx) (and (agent_holds rug ?xxx) ) ) ) (or (in_motion ?xxx rug) (and (< 1 1) (toggled_on ?xxx rug) ) ) ) )
      )
    )
    (preference preference2
      (exists (?t - (either hexagonal_bin alarm_clock block))
        (exists (?g - hexagonal_bin)
          (at-end
            (agent_holds ?t south_west_corner)
          )
        )
      )
    )
    (preference preference3
      (exists (?b - dodgeball)
        (at-end
          (in_motion ?b)
        )
      )
    )
    (preference preference4
      (then
        (hold (= (distance 3 room_center) 1) )
        (once (agent_holds ?xxx) )
        (once (agent_holds front) )
      )
    )
  )
)
(:terminal
  (> (count preference1) (total-time) )
)
(:scoring
  (count preference3:golfball)
)
)


(define (game game-id-189) (:domain many-objects-room-v1)
(:setup
  (exists (?r - (either cellphone cellphone) ?z - hexagonal_bin)
    (game-optional
      (agent_holds door)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - ball ?g - (either dodgeball cube_block) ?x ?a - ball)
        (exists (?u - (either triangle_block cube_block wall))
          (exists (?t - chair)
            (exists (?e - hexagonal_bin ?r - ball)
              (forall (?n - teddy_bear)
                (then
                  (once (and (and (adjacent ?t) (agent_holds ?x ?u) ) (adjacent ?t) (in_motion ?r ?r) ) )
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (not (agent_holds ?xxx) ) )
        (hold-while (agent_holds floor ?xxx) (on ?xxx) )
        (once (toggled_on ?xxx) )
      )
    )
    (preference preference3
      (exists (?w - ball ?m - wall)
        (then
          (once (on ?m ?m) )
          (once (not (agent_holds agent) ) )
          (once-measure (not (game_over ?m ?m) ) (distance 5 ?m desk) )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (>= 2 (count preference1:block) )
      (< 0.5 (+ 10 (count preference3:red:dodgeball) )
      )
    )
    (or
      (or
        (>= 3 1 )
        (>= (* 8 (- (count preference3) )
            (external-forall-minimize
              (* 18 (- (external-forall-minimize (* (total-time) (count preference2:beachball:beachball) 10 )
                  )
                )
                (* 5 (count preference3:blue_dodgeball) )
                (count preference3:blue)
                (count preference2:yellow:blue:brown)
              )
            )
            5
            (count-once-per-objects preference1:basketball)
          )
          (external-forall-maximize
            (not
              (count preference1:basketball:pink)
            )
          )
        )
      )
      (>= 1 (count-once-per-external-objects preference2:orange) )
      (not
        (<= 8 (+ 10 5 (- (count-longest preference1:doggie_bed:pink_dodgeball) )
          )
        )
      )
    )
    (>= (count-once-per-objects preference3:yellow:alarm_clock) 1 )
  )
)
(:scoring
  (-
    (not
      (count-once-per-external-objects preference3:alarm_clock)
    )
    (external-forall-maximize
      (count preference2:pyramid_block:dodgeball:alarm_clock)
    )
  )
)
)


(define (game game-id-190) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?w - hexagonal_bin)
      (and
        (preference preference1
          (exists (?z - block)
            (then
              (once (not (in_motion ?z ?z) ) )
              (once (in_motion pillow ?w) )
              (hold (exists (?s - doggie_bed ?q - bridge_block) (adjacent ?w ?q) ) )
              (once (on ?w) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= (= (count preference1:yellow:pink_dodgeball) (* (count-once preference1:hexagonal_bin) (total-time) )
        )
        (+ (+ (= 3 (- (count-once preference1:dodgeball:basketball) )
            )
            (<= 100 1 )
          )
          2
        )
      )
    )
  )
)
(:scoring
  6
)
)


(define (game game-id-191) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (object_orientation floor)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - dodgeball)
        (then
          (hold-for 7 (agent_holds brown desk) )
          (once (not (< (distance ?c ?c) 1) ) )
          (once (agent_holds ?c) )
        )
      )
    )
  )
)
(:terminal
  (<= 3 (* (not (* (count-overlapping preference1:cube_block) (total-score) )
      )
      8
    )
  )
)
(:scoring
  (+ 10 10 )
)
)


(define (game game-id-192) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-conserved
          (and
            (and
              (in_motion ?xxx ?xxx)
              (and
                (and
                  (and
                    (agent_holds ?xxx ?xxx)
                    (and
                      (in_motion ?xxx)
                      (not
                        (in_motion ?xxx)
                      )
                    )
                  )
                  (agent_holds ?xxx ?xxx)
                )
                (and
                  (not
                    (not
                      (>= (building_size desk ?xxx) 1)
                    )
                  )
                  (on ?xxx ?xxx)
                )
              )
              (not
                (not
                  (in_motion ?xxx ?xxx)
                )
              )
            )
            (in ?xxx ?xxx)
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
        (once (on ?xxx) )
        (once (and (exists (?t - cube_block) (and (in_motion ?t back) (not (adjacent desk ?t) ) ) ) (< (distance ?xxx room_center) 1) ) )
        (once (and (> 5 (distance 9 ?xxx)) (in_motion ?xxx) (in_motion ?xxx ?xxx) ) )
      )
    )
    (preference preference2
      (then
        (once-measure (in ?xxx ?xxx) (distance ?xxx 3) )
        (once (and (in_motion ?xxx) (and (and (not (and (agent_holds ?xxx) ) ) (and (in_motion ?xxx ?xxx) (agent_holds bed ?xxx) ) ) (same_color ?xxx ?xxx) ) (not (touch ?xxx) ) ) )
        (once (in_motion ?xxx) )
      )
    )
    (preference preference3
      (exists (?a - wall)
        (exists (?x - shelf)
          (exists (?f - pillow)
            (exists (?w ?g - game_object)
              (exists (?t - building)
                (then
                  (hold-while (not (in ?w) ) (and (and (in_motion ?t ?g) (agent_holds ?x ?g) ) (in_motion ?t) ) )
                  (once (in_motion ?t yellow) )
                  (once (in_motion ?g) )
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
  (not
    (not
      (>= 20 7 )
    )
  )
)
(:scoring
  (* (count-once-per-external-objects preference1:hexagonal_bin) (* (+ (count preference1:blue_dodgeball) 3 )
      (* 10 3 )
    )
  )
)
)


(define (game game-id-193) (:domain medium-objects-room-v1)
(:setup
  (forall (?p - dodgeball)
    (game-optional
      (or
        (and
          (and
            (on ?p)
            (not
              (on ?p)
            )
          )
          (in ?p)
        )
        (in ?p)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r ?m ?n ?u - cube_block ?l - ball)
        (then
          (once (>= 2 2) )
          (once (not (not (same_color ?l upright) ) ) )
          (hold (not (is_setup_object ?l) ) )
          (once (not (exists (?g - dodgeball ?b - hexagonal_bin) (not (in_motion ?b ?l) ) ) ) )
          (once (in_motion ?l ?l) )
          (once (agent_crouches ?l ?l) )
        )
      )
    )
    (preference preference2
      (exists (?t - cylindrical_block)
        (then
          (once (forall (?g - chair) (not (in_motion ?g ?g) ) ) )
          (once (touch ?t ?t) )
          (once (and (in_motion ?t) (agent_holds ?t ?t) ) )
        )
      )
    )
    (preference preference3
      (then
        (once (same_color ?xxx ?xxx) )
      )
    )
    (preference preference4
      (then
        (hold (agent_holds ?xxx) )
        (once (same_type ?xxx) )
        (once (not (not (and (on desk ?xxx) (not (rug_color_under ?xxx ?xxx) ) ) ) ) )
      )
    )
    (preference preference5
      (at-end
        (= (distance ?xxx 8) 1)
      )
    )
    (preference preference6
      (exists (?f - pillow)
        (exists (?n - game_object ?y - tall_cylindrical_block)
          (then
            (once (and (on ?f desk) (not (not (and (on block ?f) (not (same_color ?f) ) (agent_holds agent) ) ) ) (agent_holds floor) ) )
            (once (in_motion upright ?y) )
            (once (on bed ?y) )
          )
        )
      )
    )
  )
)
(:terminal
  (= (+ (* 7 100 )
      (count preference4:dodgeball)
    )
    1
  )
)
(:scoring
  3
)
)


(define (game game-id-194) (:domain few-objects-room-v1)
(:setup
  (exists (?f - hexagonal_bin)
    (exists (?o - hexagonal_bin ?n - hexagonal_bin)
      (game-conserved
        (on ?f ?n)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (not (and (not (in agent desk) ) (agent_holds ?xxx) (not (not (agent_holds ?xxx) ) ) ) ) (and (not (object_orientation ?xxx) ) (not (in_motion ?xxx ?xxx) ) (not (adjacent_side ?xxx) ) ) )
        (once (adjacent_side rug) )
        (once (exists (?w - drawer) (exists (?t - pyramid_block) (not (not (in ?w) ) ) ) ) )
      )
    )
    (preference preference2
      (exists (?f - ball ?j - dodgeball)
        (then
          (once (or (or (not (not (and (in_motion ?j) (not (in ?j) ) (in_motion ?j green_golfball) (and (agent_holds floor ?j) (on ?j ?j) ) ) ) ) (< 1 (distance ?j ?j)) (or (between ?j) (> 2 1) ) ) (exists (?u - dodgeball ?r - cube_block) (not (in ?r ?r) ) ) ) )
          (hold (agent_holds ?j) )
          (hold (in_motion ?j) )
        )
      )
    )
    (preference preference3
      (exists (?g - triangular_ramp)
        (exists (?f - curved_wooden_ramp)
          (then
            (once (not (= 2 (distance desk ?f)) ) )
            (hold (not (exists (?v - hexagonal_bin) (not (agent_holds yellow) ) ) ) )
            (once (in_motion agent ?f) )
            (once (not (< (distance room_center ?f) 1) ) )
            (once (and (in ?g) (and (in_motion ?f) (not (and (not (forall (?z - teddy_bear) (agent_holds door ?f) ) ) (agent_holds front brown ?f) (agent_holds ?f) ) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 18 )
)
(:scoring
  (count-once-per-objects preference2:golfball)
)
)


(define (game game-id-195) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (= 3 (distance ?xxx ?xxx))
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (equal_x_position agent ?xxx) )
        (once (and (agent_holds ?xxx) (forall (?j - rug ?a - hexagonal_bin ?c - triangular_ramp ?o - dodgeball) (in desk ?o ?o) ) (on ?xxx rug) ) )
        (hold (< 2 10) )
      )
    )
    (preference preference2
      (exists (?f - (either lamp laptop mug mug) ?c - cube_block)
        (then
          (hold (in_motion ?c ?c) )
          (hold-while (in ?c ?c) (and (not (and (on ?c ?c) (and (in_motion ?c ?c) (agent_holds ?c ?c) ) ) ) (in_motion ?c) ) (not (and (and (in_motion ?c ?c) (touch ?c ?c agent) ) (not (in ?c) ) ) ) (on ?c ?c) )
          (hold (not (in_motion ?c) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1) (count preference1:red:yellow) )
)
(:scoring
  (count preference1:hexagonal_bin)
)
)


(define (game game-id-196) (:domain medium-objects-room-v1)
(:setup
  (exists (?j - block)
    (forall (?r - blue_pyramid_block)
      (game-optional
        (in ?r)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (not (game_over agent agent) ) (not (in_motion agent) ) ) )
        (once (same_type rug ?xxx) )
        (hold-to-end (in_motion ?xxx) )
      )
    )
  )
)
(:terminal
  (> (<= (- 3 )
      (count preference1:red_pyramid_block)
    )
    (external-forall-maximize
      (/
        (count-once-per-objects preference1:beachball)
        (+ 6 (count-once preference1:hexagonal_bin) (total-score) (+ (external-forall-maximize (count preference1:basketball) ) 50 )
          (count-once-per-objects preference1:pink_dodgeball)
        )
      )
    )
  )
)
(:scoring
  (* (* (* (= (* (+ 2 (+ (+ (* (count preference1:beachball) (* (count-once preference1:pink_dodgeball) 30 )
                  )
                  (* (count-unique-positions preference1:red) (count preference1:pink_dodgeball:rug) )
                )
              )
            )
            (count-longest preference1:wall)
          )
          (* (count-once preference1:beachball) (count-once-per-external-objects preference1:dodgeball) )
          (count-once-per-objects preference1:green:golfball)
        )
        (count-once preference1:dodgeball)
      )
      (count preference1:beachball:basketball)
    )
    (* (count preference1:book) (* (count-once-per-objects preference1:basketball) (count preference1:pink:yellow) )
      (>= (count-once-per-objects preference1:hexagonal_bin) (count-once-per-external-objects preference1:green) )
    )
  )
)
)


(define (game game-id-197) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (and
      (on ?xxx rug)
      (adjacent ?xxx ?xxx)
      (and
        (not
          (adjacent ?xxx)
        )
        (not
          (and
            (and
              (in_motion desk ?xxx)
              (in_motion agent)
              (equal_z_position ?xxx ?xxx)
              (not
                (adjacent_side ?xxx ?xxx ?xxx)
              )
              (not
                (agent_holds ?xxx ?xxx)
              )
              (not
                (in_motion ?xxx desk ?xxx)
              )
              (not
                (touch ?xxx blue)
              )
              (and
                (not
                  (or
                    (and
                      (in_motion bed ?xxx)
                      (in_motion ?xxx)
                    )
                    (in ?xxx)
                  )
                )
                (on ?xxx bed)
                (not
                  (and
                    (and
                      (agent_holds ?xxx ?xxx)
                      (in ?xxx)
                    )
                    (on ?xxx ?xxx)
                    (in_motion ?xxx ?xxx)
                    (not
                      (and
                        (and
                          (and
                            (and
                              (and
                                (agent_holds bed)
                                (adjacent ?xxx)
                              )
                              (in_motion bed)
                            )
                            (and
                              (in_motion ?xxx ?xxx)
                              (in ?xxx ?xxx)
                              (agent_holds ?xxx)
                              (not
                                (and
                                  (in_motion ?xxx)
                                )
                              )
                              (on agent)
                              (and
                                (not
                                  (and
                                    (and
                                      (in ?xxx)
                                    )
                                    (agent_holds ?xxx)
                                    (agent_holds ?xxx)
                                    (adjacent ?xxx ?xxx)
                                    (< 2 6)
                                    (on ?xxx)
                                    (in_motion ?xxx ?xxx ?xxx)
                                    (not
                                      (and
                                        (and
                                          (and
                                            (same_color ?xxx)
                                            (in_motion ?xxx ?xxx)
                                          )
                                          (not
                                            (not
                                              (exists (?s - doggie_bed)
                                                (agent_holds ?s)
                                              )
                                            )
                                          )
                                        )
                                        (not
                                          (in_motion blue)
                                        )
                                      )
                                    )
                                    (in_motion green bed)
                                    (adjacent_side floor)
                                  )
                                )
                                (< (distance ?xxx front) 1)
                              )
                              (exists (?z - shelf)
                                (agent_holds left ?z)
                              )
                              (< (distance ?xxx ?xxx) 1)
                            )
                          )
                          (and
                            (on ?xxx)
                            (in blue ?xxx)
                          )
                        )
                        (not
                          (not
                            (> (distance ?xxx ?xxx) 1)
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
            (agent_holds ?xxx agent)
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
        (on ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once-per-objects preference1:golfball) (count preference1:beachball) )
    (or
      (<= 40 (* 3 (count-same-positions preference1:basketball) )
      )
    )
  )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-198) (:domain medium-objects-room-v1)
(:setup
  (forall (?b - (either desktop pyramid_block pyramid_block))
    (exists (?j - hexagonal_bin ?s - shelf ?u ?g ?l ?j ?z ?y - hexagonal_bin ?z - block)
      (or
        (game-conserved
          (adjacent_side pink ?z)
        )
        (exists (?m - golfball)
          (exists (?l ?i - curved_wooden_ramp)
            (and
              (game-optional
                (agent_holds ?l ?b)
              )
            )
          )
        )
        (exists (?a - hexagonal_bin)
          (game-conserved
            (in_motion ?b)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - hexagonal_bin ?w - (either chair cylindrical_block))
        (then
          (once (agent_holds ?w) )
          (once (in_motion ?w) )
          (once (< 7 (distance ?w ?w ?w)) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (count-once-per-objects preference1:basketball) )
)
(:scoring
  (* 3 2 )
)
)


(define (game game-id-199) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (and
      (rug_color_under brown ?xxx)
      (or
        (not
          (and
            (not
              (and
                (on ?xxx ?xxx)
                (and
                  (not
                    (not
                      (exists (?y - curved_wooden_ramp ?p - pillow)
                        (in_motion back)
                      )
                    )
                  )
                  (in_motion ?xxx ?xxx)
                )
              )
            )
            (agent_holds ?xxx bed)
          )
        )
        (in_motion pink_dodgeball)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - shelf ?j - golfball)
        (exists (?k ?b - hexagonal_bin)
          (then
            (hold (not (or (and (on upright ?b) (in ?j desk) ) (rug_color_under ?j) (not (on ?j) ) (in_motion ?b) ) ) )
            (once (and (not (not (= (distance agent agent) 1) ) ) (not (and (< (distance room_center 8 room_center) (distance agent ?j)) (and (>= (distance room_center ?b) 1) (and (agent_holds ?k ?b) (>= 9 1) ) (touch ?j agent) ) ) ) ) )
            (once (not (not (and (in agent ?k) (= 1 (distance ?k room_center)) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 0.7 )
)
(:scoring
  7
)
)


(define (game game-id-200) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (on ?xxx ?xxx) (not (and (and (< (distance ?xxx ?xxx) 4) (not (in_motion rug ?xxx) ) (in_motion ?xxx) (agent_holds ?xxx ?xxx) ) (in_motion ?xxx) ) ) ) )
      )
    )
    (preference preference2
      (exists (?t - (either pyramid_block alarm_clock))
        (exists (?w - color)
          (exists (?l - curved_wooden_ramp ?y - beachball)
            (then
              (once (agent_holds ?w ?t) )
              (hold (and (or (and (and (or (on ?y agent) (agent_holds ?w) (< 1 1) ) (in_motion ?w) ) (agent_holds ?t) ) (< 1 (distance room_center room_center)) ) (not (agent_holds ?w) ) ) )
              (hold (and (agent_holds ?w) (in_motion ?w) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preference1:purple) (count-increasing-measure preference2:pink) (count-once-per-objects preference1:brown) 0 0 )
    (count preference1:orange)
  )
)
(:scoring
  8
)
)


(define (game game-id-201) (:domain medium-objects-room-v1)
(:setup
  (forall (?a - ball ?v - game_object)
    (not
      (game-conserved
        (and
          (in_motion pillow)
          (adjacent ?v)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (in ?xxx)
      )
    )
  )
)
(:terminal
  (>= 6 (count-once-per-objects preference1:golfball) )
)
(:scoring
  (count preference1:hexagonal_bin)
)
)


(define (game game-id-202) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx rug)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (in_motion agent ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
        (hold (< 2 9) )
      )
    )
    (preference preference2
      (exists (?x - hexagonal_bin)
        (exists (?i - hexagonal_bin)
          (exists (?s - book ?k - golfball ?b ?a ?v ?e - dodgeball)
            (exists (?t - (either cylindrical_block alarm_clock))
              (exists (?m - hexagonal_bin)
                (at-end
                  (not
                    (agent_holds ?t)
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
  (>= (* (* (* (count-once-per-objects preference2:pink:green) 3 )
        (* (count preference2:basketball:pink) (* 6 (count-same-positions preference1:yellow:red) )
          (* (+ (* (external-forall-maximize (external-forall-maximize (count-once preference1:blue_cube_block) ) ) (count-unique-positions preference1:dodgeball) )
              3
            )
            (total-score)
          )
        )
      )
      (count preference2:yellow)
      (count-increasing-measure preference2:pink:red:cube_block)
    )
    (count-same-positions preference2:golfball:dodgeball:pink_dodgeball)
  )
)
(:scoring
  (+ (count-once preference2:pink:yellow) (* 3 (and 10 (+ (count-shortest preference1:golfball) )
        (+ (= (+ 4 (* (or (* (* (count-once-per-objects preference2:yellow) (count preference2:beachball) )
                    8
                  )
                )
                (count preference1:pink_dodgeball)
                (* (* (count preference2:yellow) (* 9 )
                  )
                  (count preference2:doggie_bed:purple)
                )
              )
              (count-total preference1:beachball:yellow_cube_block:red)
              (total-score)
            )
            2
          )
          (+ (* (* (total-score) )
              (total-score)
            )
            300
          )
        )
      )
    )
  )
)
)


(define (game game-id-203) (:domain medium-objects-room-v1)
(:setup
  (exists (?l - building ?f - curved_wooden_ramp)
    (forall (?z - hexagonal_bin ?p - hexagonal_bin)
      (exists (?r - ball ?o - (either bridge_block laptop) ?w - (either cd teddy_bear))
        (game-conserved
          (and
            (in_motion front ?w)
            (forall (?x - cube_block ?c - hexagonal_bin)
              (on ?c)
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
          (exists (?r - game_object ?u - dodgeball)
            (at-end
              (exists (?q - hexagonal_bin ?y - hexagonal_bin)
                (on ?g)
              )
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (same_color ?xxx) )
        (once (and (in front) (on agent) ) )
        (hold (and (and (is_setup_object ) (not (touch ?xxx) ) ) (< 1 (distance room_center ?xxx)) ) )
      )
    )
    (preference preference3
      (exists (?u - dodgeball)
        (exists (?i - hexagonal_bin ?k - dodgeball ?j - teddy_bear)
          (exists (?x - hexagonal_bin ?q - (either pyramid_block credit_card hexagonal_bin))
            (exists (?c - doggie_bed ?c - teddy_bear)
              (then
                (once (< (distance ?c) 0.5) )
                (hold (not (in ?j upside_down) ) )
                (forall-sequence (?r - hexagonal_bin ?w - ball)
                  (then
                    (once (in_motion agent) )
                    (once (equal_x_position ?u) )
                    (once (not (in_motion ?c bed) ) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference4
      (at-end
        (in_motion ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (>= (* (or 2 ) 20 )
    (count preference1)
  )
)
(:scoring
  2
)
)


(define (game game-id-204) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (touch ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (forall (?z - dodgeball ?r - (either wall yellow_cube_block))
        (then
          (hold-for 2 (and (agent_holds ?r) (on ?r) ) )
          (once (in_motion ?r) )
          (once (and (agent_holds ?r) (agent_crouches ?r) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (* (external-forall-maximize (external-forall-maximize (* (total-time) (count preference1:pink_dodgeball) )
        )
      )
      (or
        (count preference1:doggie_bed:doggie_bed)
        30
      )
    )
  )
)
(:scoring
  6
)
)


(define (game game-id-205) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (on desk ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?g ?h - dodgeball)
      (and
        (preference preference1
          (then
            (hold-while (not (and (between ?h ?g) (exists (?k - (either alarm_clock side_table yellow doggie_bed wall pillow cylindrical_block)) (in_motion ?k desk) ) ) ) (on ?h) (and (in_motion ?g ?h) (in_motion ?g) (and (and (and (object_orientation ?g ?h) (or (and (and (> 9 (distance ?h 5)) (and (agent_holds ?h ?g) (on ?h ?h) ) ) (agent_holds bed ?g) ) (not (in_motion green block) ) ) (= (x_position ?g desk) (distance 6 room_center)) (and (on ?h ?g) (on ?h right) (adjacent desk ?h) ) ) (on ?h ?g) ) (and (on ?h) (same_object agent ?h) ) ) ) )
            (once (and (and (exists (?o - doggie_bed ?z - ball) (in_motion ?h) ) (and (object_orientation front_left_corner) (on ?h agent) ) ) (on ?g pillow) ) )
            (once (not (and (agent_holds ?h) (not (not (not (and (not (exists (?y - (either dodgeball floor) ?w - triangular_ramp) (and (agent_holds ?w) (in ?w) ) ) ) (and (in_motion ?h) (not (and (touch ?h) (not (agent_holds ?h ?g) ) (not (and (touch ?h) (or (agent_holds ?g) ) ) ) (and (adjacent pink) (and (not (and (game_start ?h) (in_motion ?g ?g) ) ) (and (in_motion door) (and (in ?h) (adjacent ?h ?g) ) ) ) ) ) ) ) ) ) ) ) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (or
      (>= (count preference1:golfball:yellow) 10 )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-206) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (in_motion ?xxx ?xxx)
      )
    )
    (game-conserved
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (on ?xxx) )
        (hold (same_color bed) )
        (once-measure (and (< (distance ?xxx room_center) 2) (agent_holds agent ?xxx) ) (distance ) )
      )
    )
    (preference preference2
      (then
        (once (exists (?u - building) (and (and (in ?u) (in_motion ?u ?u) ) (in_motion agent ?u) ) ) )
        (once (in_motion ?xxx desk) )
        (once (agent_holds ) )
      )
    )
    (forall (?p - (either basketball cellphone))
      (and
        (preference preference3
          (at-end
            (agent_holds ?p ?p)
          )
        )
        (preference preference4
          (then
            (hold-while (in_motion ?p) (in ?p ?p) )
            (once (in_motion ?p) )
            (hold-while (and (not (>= (distance 5 ?p) (distance bed ?p)) ) ) (agent_holds ?p) (or (in ?p) (in ?p) ) (and (and (agent_holds ?p) (on ?p) ) (in_motion ?p ?p) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* 4 (* 2 4 )
    )
    (count-once-per-objects preference4:dodgeball)
  )
)
(:scoring
  (or
    (count preference4:golfball)
  )
)
)


(define (game game-id-207) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (forall (?n - (either dodgeball floor))
        (exists (?k ?i - (either cd bed))
          (and
            (game-conserved
              (not
                (not
                  (and
                    (and
                      (agent_holds ?i ?i)
                      (agent_holds ?k)
                    )
                    (not
                      (on ?k)
                    )
                  )
                )
              )
            )
            (and
              (and
                (game-conserved
                  (agent_crouches floor)
                )
              )
            )
          )
        )
      )
    )
    (forall (?s - wall ?g - blue_cube_block)
      (forall (?n - (either mug cube_block golfball))
        (game-conserved
          (in_motion pink_dodgeball)
        )
      )
    )
    (game-optional
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?c - tall_cylindrical_block ?o - hexagonal_bin)
      (and
        (preference preference1
          (exists (?r - ball)
            (forall (?u - wall)
              (exists (?n - dodgeball)
                (exists (?i - game_object ?v - (either pyramid_block dodgeball cellphone ball key_chain pyramid_block chair) ?l - hexagonal_bin)
                  (then
                    (once (in_motion ?l) )
                    (once (not (object_orientation ?l) ) )
                    (once (adjacent ?n) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (at-end
        (same_color ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (= (total-score) (count-increasing-measure preference1:green:dodgeball) )
    (>= 7 (count preference1:golfball:dodgeball) )
  )
)
(:scoring
  5
)
)


(define (game game-id-208) (:domain medium-objects-room-v1)
(:setup
  (or
    (game-optional
      (not
        (in_motion ?xxx ?xxx)
      )
    )
    (exists (?a - bridge_block)
      (game-optional
        (and
          (not
            (not
              (in_motion ?a)
            )
          )
          (on ?a ?a)
          (object_orientation ?a)
          (agent_holds ?a)
        )
      )
    )
    (or
      (game-optional
        (not
          (< 10 (distance 7 ?xxx))
        )
      )
      (game-conserved
        (forall (?u - (either hexagonal_bin laptop desktop))
          (not
            (agent_holds ?u)
          )
        )
      )
      (game-conserved
        (and
          (and
            (not
              (= 4 0 (distance ?xxx ?xxx) 1)
            )
            (and
              (agent_holds ?xxx desktop)
              (< (distance side_table ?xxx) 1)
            )
          )
          (not
            (and
              (and
                (and
                  (agent_holds ?xxx)
                  (agent_holds ?xxx)
                )
                (not
                  (adjacent ?xxx)
                )
              )
              (in_motion ?xxx)
              (in_motion ?xxx)
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
      (exists (?y - sliding_door ?d - hexagonal_bin ?v - (either rug doggie_bed))
        (then
          (once (on ?v) )
          (once (same_color ?v) )
          (once (between ?v ?v) )
        )
      )
    )
    (preference preference2
      (exists (?a - beachball)
        (exists (?l - hexagonal_bin ?w - teddy_bear)
          (then
            (once (in_motion ?a) )
            (once-measure (exists (?n - dodgeball) (not (agent_holds ?a) ) ) (distance ?w agent) )
          )
        )
      )
    )
  )
)
(:terminal
  (> (or (count preference2:pink:basketball) 5 (count preference1:triangle_block) ) 10 )
)
(:scoring
  100
)
)


(define (game game-id-209) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (and
      (not
        (faces ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - (either dodgeball game_object) ?f - hexagonal_bin)
        (then
          (once (adjacent_side ?f ?f) )
          (once (not (agent_holds bed rug) ) )
          (once (agent_holds ?f) )
        )
      )
    )
    (forall (?o - cube_block)
      (and
        (preference preference2
          (then
            (once (agent_holds ?o) )
            (hold (exists (?u - dodgeball ?m - hexagonal_bin) (adjacent ?m ?o) ) )
            (once (on ?o) )
          )
        )
        (preference preference3
          (exists (?z - bridge_block ?k - hexagonal_bin)
            (then
              (hold (not (on ?o ?k) ) )
              (hold (in_motion ?k agent) )
              (hold (and (object_orientation ?k) (not (in_motion ?o) ) (on bed) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (+ 1 0 )
  )
)
(:scoring
  (count preference3:yellow_cube_block)
)
)


(define (game game-id-210) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - red_pyramid_block ?a - game_object)
        (exists (?h - game_object)
          (then
            (once (in pillow) )
            (once (same_type ?a) )
            (once (agent_holds ?h) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preference1:pink_dodgeball) 10 )
)
(:scoring
  10
)
)


(define (game game-id-211) (:domain medium-objects-room-v1)
(:setup
  (exists (?x - (either cd dodgeball))
    (forall (?z - hexagonal_bin)
      (exists (?n - shelf ?i - game_object ?w ?f - curved_wooden_ramp ?k - ball ?q - block)
        (and
          (game-conserved
            (not
              (and
                (agent_holds ?x ?z)
                (not
                  (on ?z)
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
      (exists (?j ?t - chair ?f - block)
        (then
          (once (agent_holds ?f ?f) )
          (once (not (and (not (<= (distance 0 ?f) (distance ?f ?f)) ) (agent_holds ?f ?f) ) ) )
          (hold (not (in_motion ) ) )
          (hold-while (in_motion ?f ?f) (and (agent_holds pink) (and (on pink_dodgeball) (in ?f ?f) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?k - (either alarm_clock dodgeball dodgeball) ?u - teddy_bear)
        (exists (?q - hexagonal_bin)
          (exists (?b - teddy_bear)
            (then
              (once (and (in_motion ?b) (is_setup_object ?b) ) )
              (hold (exists (?w - game_object) (< 1 1) ) )
              (once (agent_holds ?u) )
            )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (and (and (in_motion ?xxx ?xxx) (not (not (> 4 1) ) ) ) (agent_holds ?xxx ?xxx) (on ) (and (not (not (exists (?m - dodgeball) (touch desk ?m) ) ) ) (adjacent ?xxx ?xxx) ) ) )
        (once (agent_holds desk ?xxx) )
        (once (in_motion ?xxx) )
      )
    )
    (preference preference4
      (then
        (hold (agent_holds ?xxx) )
        (hold (in_motion ?xxx ?xxx) )
        (once (< 1 (distance door agent)) )
      )
    )
    (preference preference5
      (then
        (hold-while (not (agent_holds ?xxx pink) ) (not (in_motion ?xxx) ) (not (<= (x_position ?xxx ?xxx) (distance 9 ?xxx)) ) )
        (once (not (and (touch ?xxx) (not (same_color ?xxx) ) ) ) )
        (once (and (and (in_motion ?xxx) (in ?xxx ?xxx) ) (and (and (in_motion ?xxx) (in_motion ?xxx) ) (not (not (and (not (same_type main_light_switch) ) (in_motion ?xxx) ) ) ) ) ) )
      )
    )
    (preference preference6
      (exists (?p - dodgeball)
        (exists (?t - doggie_bed)
          (then
            (hold (and (not (and (not (not (adjacent ?p ?t) ) ) (and (<= (distance ?t 9) (distance room_center ?p)) (agent_holds ?p) (and (in ?p ?t) (not (and (adjacent_side ?p) (in ?t bed) ) ) ) ) ) ) ) )
            (once (agent_holds ?t bed bed) )
            (hold (in_motion ?p top_drawer) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference4:green) (count-once-per-external-objects preference4:orange) )
)
(:scoring
  (count-once-per-external-objects preference1:golfball)
)
)


(define (game game-id-212) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?k - cylindrical_block)
      (not
        (game-optional
          (agent_holds ?k ?k)
        )
      )
    )
    (exists (?i - hexagonal_bin)
      (and
        (game-conserved
          (in_motion ?i)
        )
        (forall (?e - (either lamp pillow laptop) ?g - hexagonal_bin)
          (game-conserved
            (not
              (agent_holds ?g)
            )
          )
        )
        (not
          (game-optional
            (in ?i ?i)
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
        (once (adjacent_side ?xxx ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
        (hold (not (and (agent_holds rug) (< 1 (distance 9 ?xxx)) ) ) )
      )
    )
  )
)
(:terminal
  (>= (+ (- (* (* (+ (= (>= (* 10 (* (* (count preference1:yellow) (count-once-per-objects preference1:yellow) )
                      (* (total-time) 5 60 (count preference1:doggie_bed) (+ (count-once-per-objects preference1:blue_dodgeball) 7 )
                        (* 6 (count-once-per-objects preference1:dodgeball) )
                      )
                      (external-forall-maximize
                        (external-forall-maximize
                          (+ (* (+ (count-once preference1) (+ 2 )
                              )
                              7
                            )
                            (count preference1:side_table)
                          )
                        )
                      )
                      (count-once-per-objects preference1:beachball)
                      10
                      (count preference1:dodgeball)
                      (count preference1:triangle_block)
                    )
                    (* (- 2 )
                      (* 10 (- 4 )
                      )
                      (count-once-per-objects preference1:dodgeball)
                      (* 2 30 5 (* (* (count preference1:dodgeball:golfball) (* (count-once preference1:pink_dodgeball) 100 )
                          )
                          (count-once-per-external-objects preference1:yellow_cube_block)
                        )
                        (count-once-per-external-objects preference1:red_pyramid_block)
                        (count-once-per-objects preference1:dodgeball)
                        6
                        4
                        (count-once-per-objects preference1:book)
                      )
                    )
                    0
                    (count-once-per-objects preference1:tall_cylindrical_block:dodgeball)
                    (-
                      (count preference1:green)
                      10
                    )
                    (* (not (* (* (count-once-per-objects preference1:dodgeball) 5 )
                          (* (+ (* (count-once-per-objects preference1) (external-forall-maximize (- (* (* 5 (* (and (count-same-positions preference1:golfball:dodgeball) ) (count-once-per-external-objects preference1:golfball) (* (count-once-per-objects preference1:blue_dodgeball) (count preference1:yellow) )
                                        )
                                        1
                                      )
                                      (= (+ (count preference1:yellow_pyramid_block) 1 )
                                        (count-once-per-objects preference1:green)
                                      )
                                    )
                                  )
                                )
                              )
                              (* (count-once preference1:pink) (- 6 (count-once-per-objects preference1:hexagonal_bin) ) (count preference1:yellow) (count preference1:pink) (count preference1:pink_dodgeball) (or (count-increasing-measure preference1:yellow) (count-once-per-objects preference1:pink_dodgeball) ) )
                            )
                            (count preference1:green:basketball)
                            3
                          )
                        )
                      )
                      (+ (count-longest preference1:beachball:doggie_bed) 3 )
                    )
                    (count-once-per-objects preference1:hexagonal_bin)
                    (count-once preference1:top_drawer)
                    3
                    (+ (count preference1:hexagonal_bin) (- 3 )
                    )
                    (count-once preference1:beachball)
                  )
                  3
                )
                (count-increasing-measure preference1:red)
              )
              (count-once-per-objects preference1:brown)
            )
            (count-once-per-objects preference1:basketball:dodgeball:dodgeball)
          )
          (external-forall-maximize
            (total-score)
          )
        )
      )
      (-
        (count preference1:triangle_block)
        (+ 2 )
      )
    )
    (* (count preference1:golfball) (count-once-per-objects preference1:dodgeball) )
  )
)
(:scoring
  3
)
)


(define (game game-id-213) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx) )
        (hold (rug_color_under ?xxx) )
        (once (and (forall (?t - hexagonal_bin) (on ?t) ) (not (agent_holds agent ?xxx) ) (and (< (distance 10 room_center) (distance ?xxx agent)) (not (and (on desk ?xxx) (and (agent_holds ?xxx) (< (distance ?xxx) 1) ) ) ) (on ?xxx) ) ) )
      )
    )
    (preference preference2
      (then
        (once (on ?xxx) )
        (hold (agent_holds ?xxx ?xxx) )
        (hold (exists (?s - building) (and (in_motion pink_dodgeball ?s) (in_motion ?s ?s) ) ) )
      )
    )
  )
)
(:terminal
  (and
    (or
      (or
        (>= (* (external-forall-maximize 10 ) 18 )
          (* (- (* (+ (count-once preference1:basketball) (external-forall-minimize (* 10 )
                  )
                  1
                )
                (/
                  (count preference2:basketball:doggie_bed)
                  (* 40 9 )
                )
                (* (count preference2:purple:pink_dodgeball) (count-once-per-objects preference1:doggie_bed) )
                5
                (count preference1:dodgeball:beachball)
                (+ (* (count preference1:golfball:hexagonal_bin) (count-once-per-objects preference1:dodgeball:dodgeball) )
                  (-
                    (count preference1:side_table)
                    0
                  )
                )
              )
            )
            (count preference2:blue_cube_block)
          )
        )
        (and
          (or
            (>= (count preference1:pink_dodgeball:white) (count preference2:purple) )
            (not
              (>= (count preference2:golfball) (count-unique-positions preference2:blue_dodgeball) )
            )
          )
          (>= (+ 5 (* 10 6 )
            )
            (count preference2:beachball)
          )
          (or
            (<= (= (* (or 10 ) (- (count-once-per-objects preference1:pink_dodgeball:beachball) )
                  (count preference2:white:dodgeball:golfball)
                  2
                  100
                  (count preference2:basketball)
                  15
                  (count preference2:side_table:dodgeball)
                  (count-once-per-objects preference1:dodgeball)
                )
                (+ (* (* (count-shortest preference2:dodgeball) 2 )
                    (count preference2:dodgeball:beachball)
                  )
                  3
                )
              )
              10
            )
            (>= (* (* (* 3 (- (* (+ (* (count-unique-positions preference1:beachball) (count-once-per-objects preference2:alarm_clock) )
                          (= 5 )
                        )
                        (* (count preference1:dodgeball:basketball) (external-forall-maximize (count preference2:blue_dodgeball) ) )
                      )
                    )
                  )
                  (total-score)
                )
                (count-once-per-objects preference1:dodgeball)
                (external-forall-maximize
                  (count preference2:yellow)
                )
              )
              (count preference1:basketball:pink_dodgeball)
            )
            (>= (* 10 (count-once preference2:pink) )
              (+ 3 10 )
            )
          )
        )
        (>= (count preference2:pink:basketball) (* (count preference2:basketball) (* (count preference2:tall_cylindrical_block) )
          )
        )
      )
      (and
        (>= (count preference2:doggie_bed:purple) (+ (* 3 (count preference2:golfball) )
            (* (count preference2:top_drawer) (count-longest preference1:yellow) )
          )
        )
      )
    )
    (>= (count preference2:golfball) (count preference1:pink_dodgeball:pink) )
    (<= (+ (+ (count-once-per-objects preference1:tan) 3 )
        2
      )
      (* (count preference1:golfball) 6 )
    )
  )
)
(:scoring
  (= 2 (count preference2:yellow_cube_block) )
)
)


(define (game game-id-214) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (exists (?z - dodgeball)
      (and
        (and
          (agent_holds agent)
          (and
            (not
              (in_motion ?z ?z)
            )
            (agent_holds ?z ?z)
            (in_motion ?z)
          )
        )
        (and
          (in_motion ?z)
          (game_over rug)
          (in_motion ?z bed)
          (agent_holds rug ?z)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx) )
        (hold (not (and (< 4 1) (in ?xxx) ) ) )
        (once (in_motion ?xxx) )
      )
    )
    (preference preference2
      (exists (?j - hexagonal_bin)
        (exists (?p - dodgeball)
          (exists (?b - chair ?q ?e - beachball)
            (exists (?y - dodgeball ?n ?b - (either ball beachball red))
              (exists (?w - block)
                (exists (?t - dodgeball)
                  (then
                    (once (in_motion bed) )
                    (once (not (not (and (< 1 (distance ?q desk)) (on ?b) ) ) ) )
                    (once (touch ?n pink_dodgeball) )
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
    (not
      (>= 30 (* 1 7 5 3 )
      )
    )
    (= (count preference1:yellow_cube_block) 8 )
  )
)
(:scoring
  3
)
)


(define (game game-id-215) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?l - dodgeball)
      (game-optional
        (not
          (adjacent_side desk top_drawer)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?w - hexagonal_bin ?e - (either golfball wall))
      (and
        (preference preference1
          (then
            (once (not (agent_holds ?e ?e) ) )
            (once (on ?e) )
            (hold (agent_holds ?e) )
          )
        )
        (preference preference2
          (then
            (once (not (not (agent_holds ?e ?e) ) ) )
            (hold (agent_holds ?e) )
            (hold (in_motion ?e) )
          )
        )
        (preference preference3
          (then
            (hold (on ?e) )
            (once (touch ?e) )
            (once (agent_holds agent) )
          )
        )
      )
    )
    (preference preference4
      (exists (?n - ball)
        (at-end
          (and
            (agent_holds ?n)
            (and
              (agent_holds ?n)
              (agent_holds agent)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:doggie_bed) (count-once-per-objects preference2:basketball:dodgeball) )
)
(:scoring
  (* (count-once preference3:basketball) (* (* 3 30 )
      1
    )
  )
)
)


(define (game game-id-216) (:domain many-objects-room-v1)
(:setup
  (forall (?x ?k - (either golfball key_chain))
    (game-conserved
      (agent_holds ?k)
    )
  )
)
(:constraints
  (and
    (forall (?s - sliding_door ?o - cube_block ?a - (either cylindrical_block cylindrical_block desktop cube_block rug cd alarm_clock) ?v - cube_block ?h - chair)
      (and
        (preference preference1
          (then
            (once (agent_holds ?h ?h) )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference1:dodgeball) (- (count-once-per-objects preference1:beachball) )
  )
)
(:scoring
  3
)
)


(define (game game-id-217) (:domain many-objects-room-v1)
(:setup
  (exists (?j - hexagonal_bin)
    (game-optional
      (<= 1 (distance 4 ?j))
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o ?x - golfball ?l - (either key_chain cube_block) ?m - dodgeball ?y - hexagonal_bin)
        (then
          (hold (not (not (not (exists (?m - dodgeball ?h - ball ?t - game_object) (and (in_motion ?y) ) ) ) ) ) )
          (hold-for 4 (agent_holds ?y) )
          (once (not (in ?y ?y) ) )
          (once (adjacent ?y ?y) )
          (hold (in ?y ?y) )
          (once (not (in_motion ?y ?y) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) (+ 10 (count preference1:rug) )
  )
)
(:scoring
  (* (+ 1 (external-forall-minimize 6 ) (* (- (* (and (- (count preference1:green) )
            )
            (* 6 (+ (count-once-per-objects preference1:beachball:yellow) (* (* (total-score) (count-increasing-measure preference1:yellow_cube_block) (count-unique-positions preference1:green:dodgeball) )
                  (< (count preference1:dodgeball:doggie_bed) (* (count preference1:golfball) (+ (* (count-once preference1:basketball:dodgeball) (count-longest preference1:golfball:green) )
                        (count-unique-positions preference1:dodgeball)
                      )
                    )
                  )
                  300
                )
              )
            )
          )
        )
        (* 3 )
      )
      (count-once-per-objects preference1:hexagonal_bin)
      (* (* (* (count preference1:pink_dodgeball:wall) 15 )
          (count-increasing-measure preference1:yellow)
        )
        (- 8 )
      )
      (count-same-positions preference1:blue_pyramid_block)
    )
    (-
      (* (count-once-per-objects preference1:beachball:yellow) (count-once-per-objects preference1:yellow) )
      (external-forall-minimize
        10
      )
    )
  )
)
)


(define (game game-id-218) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (not
          (and
            (agent_holds ?xxx ?xxx)
            (agent_holds ?xxx)
          )
        )
        (and
          (agent_holds ?xxx)
          (and
            (not
              (not
                (in ?xxx ?xxx)
              )
            )
            (in_motion ?xxx ?xxx)
            (and
              (in_motion ?xxx ?xxx ?xxx)
              (and
                (agent_holds ?xxx ?xxx)
                (< 1 (building_size ?xxx 3))
              )
            )
            (agent_holds ?xxx)
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
            (once (same_color ?c) )
            (hold (exists (?w - pillow) (< (distance room_center ?w) (distance )) ) )
            (once (not (not (< 1 4) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:yellow) 20 )
)
(:scoring
  (= 1 3 )
)
)


(define (game game-id-219) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (on ?xxx ?xxx)
        (agent_crouches ?xxx rug)
      )
    )
    (game-conserved
      (toggled_on ?xxx)
    )
    (game-optional
      (not
        (forall (?f - hexagonal_bin)
          (not
            (agent_holds ?f ?f)
          )
        )
      )
    )
    (forall (?f - golfball)
      (game-conserved
        (object_orientation ?f)
      )
    )
    (exists (?i - (either pen desktop cellphone))
      (exists (?m - (either cellphone ball chair))
        (game-conserved
          (and
            (agent_holds ?i ?m)
            (not
              (in_motion ?m)
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
        (once (agent_holds ?xxx) )
        (hold-for 3 (and (and (agent_holds bed) (in_motion ?xxx) ) (not (and (and (forall (?y - hexagonal_bin) (in_motion ?y) ) (in ?xxx) ) (in_motion ?xxx) ) ) ) )
        (once (in_motion ?xxx rug) )
      )
    )
  )
)
(:terminal
  (>= 10 (+ (count-once-per-objects preference1:dodgeball) (* 2 1 )
    )
  )
)
(:scoring
  (external-forall-maximize
    5
  )
)
)


(define (game game-id-220) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (not
        (in_motion ?xxx)
      )
      (< 4 2)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (on ?xxx)
      )
    )
  )
)
(:terminal
  (>= 50 (+ (count-once-per-objects preference1:green:dodgeball) (count preference1:orange) )
  )
)
(:scoring
  (count-unique-positions preference1:blue_pyramid_block)
)
)


(define (game game-id-221) (:domain many-objects-room-v1)
(:setup
  (or
    (game-conserved
      (rug_color_under ?xxx)
    )
    (exists (?m - ball)
      (and
        (game-conserved
          (not
            (not
              (adjacent ?m desk)
            )
          )
        )
        (game-optional
          (and
            (in_motion ?m)
            (adjacent ?m)
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
          (and
            (is_setup_object ?xxx)
            (agent_holds ?xxx)
          )
        )
      )
    )
  )
)
(:terminal
  (<= 1 30 )
)
(:scoring
  (count-once preference1:golfball)
)
)


(define (game game-id-222) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?u - red_dodgeball)
      (forall (?p - rug)
        (exists (?k - (either floor pen main_light_switch))
          (forall (?e - (either dodgeball cube_block))
            (exists (?c - triangular_ramp ?l - (either golfball dodgeball) ?o - hexagonal_bin)
              (or
                (forall (?t ?n ?q - dodgeball)
                  (and
                    (forall (?w - (either game_object golfball laptop))
                      (exists (?m - game_object)
                        (game-conserved
                          (agent_holds ?w desk)
                        )
                      )
                    )
                    (exists (?f - hexagonal_bin ?h - hexagonal_bin)
                      (game-conserved
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
  )
)
(:constraints
  (and
    (forall (?i - (either desktop book))
      (and
        (preference preference1
          (then
            (once (forall (?h - dodgeball ?j - (either watch basketball hexagonal_bin mug) ?j ?l - game_object ?a - cube_block) (exists (?l - curved_wooden_ramp ?y - building) (and (same_color ?i ?a) (in_motion ?a ?a) ) ) ) )
            (once (agent_holds desk) )
            (hold (agent_holds ?i) )
          )
        )
      )
    )
    (preference preference2
      (exists (?n ?h - dodgeball)
        (then
          (hold (or (and (and (and (and (exists (?e ?r - triangular_ramp) (exists (?g - flat_block ?k - color ?q - (either basketball top_drawer game_object) ?o - (either pink yellow_cube_block cube_block)) (in_motion ?n) ) ) (not (agent_holds ?h) ) ) (and (agent_holds ?n ?n) (< 3 1) ) ) (and (not (agent_holds ?n) ) (and (in_motion pink_dodgeball) ) ) ) (on ?h) ) (in_motion ?h) ) )
          (hold (on ?n ?h) )
          (hold (agent_holds ?n) )
          (once (not (agent_crouches agent ?n) ) )
        )
      )
    )
  )
)
(:terminal
  (> 4 (+ (and (total-time) (count preference1:yellow) ) 0 (count-once-per-objects preference1:alarm_clock) (count preference1:alarm_clock) )
  )
)
(:scoring
  (* 2 (* (* (+ 8 (count preference1:basketball:basketball) )
        (* (* (total-time) (* (* (count preference2:purple) (count-once-per-objects preference1:dodgeball) )
              (count-once-per-objects preference2:green)
            )
          )
          (or
            (* 8 3 )
          )
        )
        (* 2 (* 6 (count preference2:beachball:pink) )
        )
        6
        (count preference2:golfball:golfball)
        (count-once-per-external-objects preference1:pink:beachball)
        (count preference1:dodgeball:yellow)
        (* (count preference2:pink_dodgeball) (+ (= (count-once-per-objects preference1:green) (count preference2:golfball:red) )
            (* 10 4 )
            6
          )
          (* (count preference2:dodgeball:dodgeball) )
        )
        (count preference1:red:bed)
      )
    )
  )
)
)


(define (game game-id-223) (:domain few-objects-room-v1)
(:setup
  (exists (?z - cube_block)
    (and
      (and
        (game-conserved
          (in_motion ?z ?z ?z)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - game_object)
        (then
          (hold (opposite ?u) )
          (once (agent_holds ?u ?u) )
          (once (agent_holds ?u) )
        )
      )
    )
    (preference preference2
      (then
        (once (not (in_motion ?xxx ?xxx) ) )
        (once (forall (?j - hexagonal_bin) (and (and (and (and (adjacent ?j) (touch ?j ?j ?j) ) (not (not (exists (?a - hexagonal_bin) (agent_holds agent) ) ) ) ) (on desk) ) (not (and (not (agent_holds bed ?j) ) (not (in_motion ?j ?j) ) ) ) ) ) )
        (once (in_motion bridge_block ?xxx upright) )
      )
    )
    (preference preference3
      (exists (?u - building)
        (exists (?v - game_object ?o ?g ?e ?y - chair ?z - (either yellow book) ?x - ball)
          (exists (?h - block ?c - dodgeball)
            (exists (?m ?d ?y - cube_block)
              (exists (?n - desk_shelf)
                (then
                  (hold (not (< (distance 2) 2) ) )
                  (once (not (on pillow) ) )
                  (once (in_motion ?u green) )
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
  (>= (* 20 4 )
    (* (* (count preference2:purple) 10 )
      (* (count preference1:doggie_bed) (* (count-once preference1:alarm_clock:yellow_cube_block) (count-once-per-objects preference3:block) )
      )
      (count preference3:dodgeball:dodgeball:golfball)
      5
    )
  )
)
(:scoring
  (* (count-once-per-external-objects preference3:green) (count-once-per-objects preference2:yellow) )
)
)


(define (game game-id-224) (:domain medium-objects-room-v1)
(:setup
  (forall (?l ?e ?s ?y - chair ?q - hexagonal_bin)
    (game-conserved
      (and
        (in ?q)
        (in_motion ?q)
        (not
          (not
            (agent_holds ?q)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - game_object)
        (exists (?p - (either basketball golfball cd))
          (then
            (hold (and (on ?q) (and (in_motion ?p) (in_motion bed) ) ) )
            (hold-while (not (< (distance front) (distance ?q ?q)) ) (in_motion ?q agent) )
            (once (and (not (and (in ?q) ) ) (not (agent_holds ?q ?q) ) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (= 5 (count preference1:hexagonal_bin) )
)
(:scoring
  (count preference1:yellow)
)
)


(define (game game-id-225) (:domain many-objects-room-v1)
(:setup
  (forall (?k ?j ?p - game_object)
    (game-optional
      (< (distance ?k) (distance ?p 7))
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (in_motion ?xxx) )
        (hold (and (same_color desk) (not (not (in_motion ?xxx) ) ) (agent_holds ?xxx ?xxx) ) )
        (forall-sequence (?d - teddy_bear)
          (then
            (once (in_motion ?d) )
            (hold (on ?d ?d) )
            (hold (agent_holds ?d) )
          )
        )
      )
    )
    (preference preference2
      (then
        (hold-while (not (and (not (or (< 1 (distance room_center agent)) (agent_holds ?xxx ?xxx) ) ) (agent_holds ?xxx) ) ) (in_motion ?xxx) )
        (hold (in ?xxx) )
        (once (on ?xxx) )
      )
    )
    (preference preference3
      (exists (?k - cube_block)
        (exists (?p - wall)
          (exists (?h ?u ?b ?o ?t ?f - cube_block ?b - (either bridge_block game_object desktop doggie_bed))
            (at-end
              (not
                (and
                  (agent_holds ?p)
                  (not
                    (agent_holds upright ?k)
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference4
      (at-end
        (not
          (in_motion ?xxx ?xxx)
        )
      )
    )
    (forall (?t - game_object)
      (and
        (preference preference5
          (exists (?o - cube_block ?o - teddy_bear)
            (exists (?w - hexagonal_bin)
              (forall (?m - hexagonal_bin)
                (exists (?e - dodgeball ?j - hexagonal_bin)
                  (exists (?i - (either laptop dodgeball))
                    (then
                      (once (not (in_motion ?m) ) )
                      (hold (not (adjacent ?j rug) ) )
                      (once (on ?m ?i) )
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
  (>= (- (+ 8 3 )
      10
    )
    (count-overlapping preference4:beachball)
  )
)
(:scoring
  7
)
)


(define (game game-id-226) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (game-optional
        (< (distance ?xxx ?xxx) 1)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (and
          (on ?xxx ?xxx)
          (agent_holds ?xxx)
          (in_motion ?xxx)
        )
      )
    )
    (preference preference2
      (exists (?b - dodgeball ?t - game_object)
        (then
          (once (in ?t) )
          (hold (agent_holds ?t) )
          (once (agent_holds ?t) )
        )
      )
    )
    (forall (?s ?g - teddy_bear)
      (and
        (preference preference3
          (then
            (hold-while (and (agent_holds ?s ?g) (forall (?m - building) (agent_holds ?m) ) ) (not (and (exists (?b - hexagonal_bin) (in_motion ?g) ) (not (in_motion ?g) ) (or (not (agent_holds ?s) ) (in_motion ?s) (and (and (not (adjacent ?g) ) (not (not (and (not (not (in ?g ?s) ) ) (in agent ?g ?s) ) ) ) ) (agent_holds ?g) ) (agent_holds ?g) ) (agent_holds ?s) ) ) (and (agent_holds ) (in_motion ?s ?s) ) (in_motion ?g) )
            (once-measure (and (same_object ?g) (in ?g ?s) ) (distance desk ?s) )
            (hold-for 9 (in_motion ?s) )
            (once (in_motion ?s desk) )
            (once (not (on ?g ?g) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once preference3:blue:golfball) 10 )
    (>= (* (external-forall-maximize (count preference2:side_table:yellow) ) (count preference1:basketball) )
      (count preference2:dodgeball)
    )
  )
)
(:scoring
  (count preference3:dodgeball)
)
)


(define (game game-id-227) (:domain few-objects-room-v1)
(:setup
  (forall (?c - (either mug book))
    (game-conserved
      (on ?c)
    )
  )
)
(:constraints
  (and
    (forall (?f - (either cylindrical_block basketball triangle_block))
      (and
        (preference preference1
          (at-end
            (not
              (is_setup_object drawer ?f)
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (forall (?p - dodgeball) (agent_holds ?p) ) )
        (once (agent_holds ?xxx) )
        (once (or (on ?xxx ?xxx) (agent_holds agent) (adjacent agent) ) )
        (once (and (and (in_motion ?xxx) (in_motion ?xxx bed) ) (on ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (* (= (total-time) (count-once-per-objects preference1:golfball) )
      (* 5 (count-once-per-objects preference2:pyramid_block) )
      10
    )
    4
  )
)
(:scoring
  (count-once-per-objects preference2:beachball:beachball)
)
)


(define (game game-id-228) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (equal_z_position ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - game_object)
        (exists (?g - game_object ?o - hexagonal_bin)
          (exists (?i - (either golfball triangle_block cube_block))
            (then
              (hold (in_motion ?k) )
              (hold (not (not (exists (?q - book ?v - cube_block) (above ?v) ) ) ) )
              (once (and (not (and (not (agent_holds ?i ?o) ) (and (or (same_color ?i) (not (agent_holds agent) ) ) (and (in_motion ?i ?i) (in ?o ?i) (agent_holds ?i) (on ?o ?o) ) (agent_holds ?i ?o ?o) ) ) ) (in_motion ?o ?k) (and (touch pink_dodgeball) (in ?o ?k) (on ?i) ) (exists (?h - hexagonal_bin) (on upright) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:yellow) (* 18 (count-once preference1:yellow:dodgeball) )
  )
)
(:scoring
  (count preference1:basketball)
)
)


(define (game game-id-229) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (and
        (exists (?u - drawer)
          (exists (?j - game_object ?c - dodgeball)
            (game-conserved
              (touch ?c)
            )
          )
        )
        (game-conserved
          (in ?xxx)
        )
      )
      (game-optional
        (< (distance ?xxx door) 2)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - game_object ?u - curved_wooden_ramp)
        (then
          (once (in_motion ?u ?u) )
          (once (and (not (in_motion ?u ?u) ) (agent_holds desk ?u) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (= (count preference1:pink) 4 )
      (* 300 5 )
      (not
        15
      )
    )
    (count-once-per-objects preference1:beachball)
  )
)
(:scoring
  5
)
)


(define (game game-id-230) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (and
      (agent_holds rug desk)
      (and
        (not
          (and
            (above pink_dodgeball ?xxx)
            (agent_holds ?xxx ?xxx)
          )
        )
        (not
          (and
            (in_motion ?xxx)
            (exists (?z - desk_shelf)
              (in agent ?z)
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
        (hold (adjacent ?xxx) )
        (once (agent_holds ?xxx agent) )
        (once (and (on ?xxx) (agent_holds ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (* (* (count preference1:golfball) (count preference1:pink) )
      (+ 3 (* 12 (* 3 (count-once-per-objects preference1:blue_dodgeball) )
        )
      )
    )
    (count-overlapping preference1:red)
  )
)
(:scoring
  (count-total preference1:pink)
)
)


(define (game game-id-231) (:domain many-objects-room-v1)
(:setup
  (exists (?q - chair)
    (game-optional
      (in ?q)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - block ?b - building)
        (exists (?l - dodgeball ?n - ball ?j - curved_wooden_ramp ?l - doggie_bed)
          (exists (?v - game_object)
            (then
              (hold (touch ?v) )
              (hold (agent_holds ?b ?v) )
              (hold-while (not (and (adjacent ?b ?v) (in_motion upright) ) ) (agent_holds ?v ?l) )
            )
          )
        )
      )
    )
    (forall (?f - (either dodgeball side_table game_object))
      (and
        (preference preference2
          (forall (?v - block)
            (then
              (once (agent_holds left ?v) )
              (once (in green) )
              (once (not (equal_z_position ?f) ) )
            )
          )
        )
        (preference preference3
          (exists (?w - block ?b - hexagonal_bin)
            (exists (?e - (either pyramid_block golfball floor))
              (exists (?j - game_object)
                (then
                  (once (in ?e ?j) )
                  (once (and (not (in ?e ?f) ) (and (and (not (rug_color_under ?j ?j) ) (or (not (not (on ?f) ) ) (agent_holds ?f) ) ) (and (in_motion ?e ?e) (same_color ?b) ) ) ) )
                  (once (game_start ?b) )
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
  (> (count preference1:golfball) (count preference3:dodgeball:hexagonal_bin) )
)
(:scoring
  18
)
)


(define (game game-id-232) (:domain many-objects-room-v1)
(:setup
  (exists (?w - curved_wooden_ramp ?d - chair)
    (exists (?z - (either yellow wall))
      (exists (?w - teddy_bear)
        (game-optional
          (adjacent ?z)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?s - chair)
      (and
        (preference preference1
          (then
            (once (adjacent_side ?s ?s) )
            (forall-sequence (?l - chair)
              (then
                (hold (and (same_color ?s ?s) (in_motion ?l sideways) ) )
                (hold (exists (?i - teddy_bear) (and (and (adjacent ?i) (in_motion ?i west_wall) ) ) ) )
                (once (not (not (faces ?l) ) ) )
              )
            )
            (any)
          )
        )
        (preference preference2
          (at-end
            (broken ?s ?s)
          )
        )
        (preference preference3
          (then
            (hold (and (on ?s ?s) (agent_holds desk) ) )
            (hold (above ?s) )
            (hold-while (adjacent_side ?s floor) (in_motion ?s) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (* 8 (* (total-score) (total-score) )
      )
      (* (count-once-per-objects preference1:beachball) (- (count-once-per-objects preference1:golfball) )
      )
    )
    (count-measure preference3:blue_pyramid_block)
  )
)
(:scoring
  (count-once-per-objects preference2:dodgeball)
)
)


(define (game game-id-233) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (forall (?l - dodgeball ?r - hexagonal_bin)
      (and
        (preference preference1
          (then
            (hold (and (agent_holds ?r) (agent_holds ?r ?r) (in ?r ?r rug) (agent_holds floor) (on ?r) (and (and (not (and (not (on ?r) ) (not (in_motion ?r ?r) ) (adjacent ?r ?r) ) ) (not (in ?r) ) (and (and (on ?r) (and (not (< 1 1) ) (and (agent_holds ?r door) (and (touch bed ?r) (and (and (agent_holds ?r) (not (in_motion bed) ) ) (on ?r rug) ) ) ) ) ) (and (not (agent_holds ?r) ) (not (touch ?r ?r) ) (on desk agent) ) (adjacent pink_dodgeball ?r) ) (and (and (agent_holds ?r ?r) (not (on ?r ?r) ) ) (in_motion ?r) ) (in_motion ?r agent) (in_motion agent) ) (in ?r) ) (and (agent_holds ?r) ) (touch ?r ?r) ) )
            (hold (on ?r) )
            (once (not (agent_holds ?r ?r) ) )
          )
        )
        (preference preference2
          (then
            (once (agent_holds rug) )
            (hold (not (same_color ?r ?r) ) )
            (hold (in_motion ?r ?r) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference2:yellow:yellow) (count preference2:doggie_bed:pink_dodgeball) )
)
(:scoring
  (count-once-per-objects preference1)
)
)


(define (game game-id-234) (:domain many-objects-room-v1)
(:setup
  (or
    (game-optional
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?i - curved_wooden_ramp)
      (and
        (preference preference1
          (then
            (once (in_motion ?i) )
            (hold (and (not (same_color ?i) ) (adjacent agent) ) )
            (once (on ?i bed) )
          )
        )
      )
    )
    (forall (?p - chair)
      (and
        (preference preference2
          (exists (?n - hexagonal_bin)
            (then
              (hold (agent_holds ?n) )
              (once (same_type ?p ?n) )
              (hold-while (between ?p) (agent_holds pink_dodgeball) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (- (count preference2:blue_pyramid_block:dodgeball) )
    3
  )
)
(:scoring
  10
)
)


(define (game game-id-235) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (exists (?l - game_object ?t ?w - (either golfball cylindrical_block))
      (in_motion ?w ?w)
    )
  )
)
(:constraints
  (and
    (forall (?q - chair)
      (and
        (preference preference1
          (then
            (once (agent_holds agent) )
          )
        )
        (preference preference2
          (then
            (once (< (distance agent ?q) 7) )
            (once (touch ?q) )
            (once (adjacent ?q ?q) )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (and (and (same_color ?xxx agent) (touch ?xxx ?xxx) ) (same_object ?xxx ?xxx) (not (and (touch ?xxx ?xxx) (in_motion ?xxx ?xxx) ) ) ) )
        (hold (on ?xxx) )
        (hold (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (and
      (or
        (>= (* 8 3 )
          (count preference3:orange)
        )
        (>= (count preference1:golfball) 5 )
        (not
          (> 1 8 )
        )
      )
    )
    (or
      (>= 6 3 )
      (> (count-once preference1:pink:golfball:dodgeball) 2 )
      (>= (+ (- (total-score) )
          (count-once preference3:pink:dodgeball:pink_dodgeball)
        )
        2
      )
    )
  )
)
(:scoring
  (+ 3 (or (> (* 50 (count preference2:beachball) (count-once-per-objects preference3:golfball) 10 (+ (* (* (count preference1:hexagonal_bin) (+ 1 2 )
              )
              (count preference3:dodgeball)
            )
          )
          (- (* (count-once-per-objects preference1:basketball) (* (count preference2) (* (* (external-forall-minimize (count preference2:pink_dodgeball) ) (count-once-per-objects preference1:hexagonal_bin) )
                  (count preference2:basketball)
                )
              )
              (count preference1:dodgeball)
              (* 6 (count-same-positions preference1:yellow) )
            )
          )
        )
        1
      )
      (count-overlapping preference2:basketball)
    )
  )
)
)


(define (game game-id-236) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (agent_holds rug ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (< 9 (distance ?xxx ?xxx))
      )
    )
  )
)
(:terminal
  (>= (<= (count-total preference1:golfball) (count-once-per-objects preference1:pink:pink_dodgeball) )
    (+ (count preference1:hexagonal_bin) (count-same-positions preference1:white) )
  )
)
(:scoring
  5
)
)


(define (game game-id-237) (:domain many-objects-room-v1)
(:setup
  (forall (?d - yellow_cube_block)
    (forall (?a ?z - hexagonal_bin ?e - hexagonal_bin)
      (forall (?h - doggie_bed)
        (or
          (game-optional
            (and
              (agent_holds upright)
              (in tan ?e)
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
        (touch ?xxx)
      )
    )
  )
)
(:terminal
  (> (* (count preference1:blue_pyramid_block) 5 )
    7
  )
)
(:scoring
  8
)
)


(define (game game-id-238) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (adjacent ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - (either pillow golfball triangular_ramp) ?o - (either lamp alarm_clock triangle_block))
        (exists (?v - hexagonal_bin)
          (then
            (once (not (same_color ?o) ) )
            (once (in rug) )
            (hold (not (>= (distance ?v room_center) 1) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (< 60 3 )
)
(:scoring
  (* (- (* (count preference1:golfball:block) 4 )
    )
    (count preference1:beachball)
  )
)
)


(define (game game-id-239) (:domain few-objects-room-v1)
(:setup
  (exists (?n ?w ?y ?u ?g ?d - triangular_ramp)
    (game-optional
      (or
        (agent_holds ?y)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (and (and (in ?xxx ?xxx) (agent_holds ?xxx) ) (not (adjacent ?xxx) ) ) )
        (hold (not (and (touch ?xxx) (on ?xxx ?xxx) ) ) )
        (forall-sequence (?d - pillow)
          (then
            (hold (in_motion ?d) )
            (once (not (on ?d ?d) ) )
            (once (agent_holds agent ?d) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (total-score) )
)
(:scoring
  (* 6 (external-forall-maximize (count preference1:triangle_block) ) )
)
)


(define (game game-id-240) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (not
      (> 1 (distance 7 ?xxx))
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (on ?xxx) )
        (hold (in_motion desk ?xxx) )
        (hold (agent_holds ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:dodgeball:hexagonal_bin) 1 )
)
(:scoring
  0
)
)


(define (game game-id-241) (:domain few-objects-room-v1)
(:setup
  (exists (?b - block)
    (and
      (game-conserved
        (in ?b)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (< (distance_side 6 green_golfball) 1) ) )
        (once (in_motion ?xxx ?xxx) )
        (once (agent_holds ?xxx ?xxx) )
      )
    )
    (preference preference2
      (exists (?y - yellow_cube_block)
        (forall (?v - block)
          (exists (?d ?z ?w - cylindrical_block)
            (exists (?x - hexagonal_bin)
              (exists (?q - doggie_bed)
                (exists (?h - (either dodgeball) ?o - cube_block ?n - (either hexagonal_bin chair))
                  (then
                    (once (on ?q ?n) )
                    (once (agent_holds ?v) )
                    (once (not (agent_holds floor) ) )
                  )
                )
              )
            )
          )
        )
      )
    )
    (forall (?g - shelf ?t - cube_block)
      (and
        (preference preference3
          (exists (?a - (either hexagonal_bin block))
            (then
              (hold (touch ?a) )
              (once (not (not (and (or (on ?a) (same_color ?t ?a) ) (in_motion ?t) ) ) ) )
              (once (in_motion desk) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count-once-per-objects preference3:beachball) 5 )
    (>= (count preference2:dodgeball:dodgeball) 2 )
    (>= 10 (* (* (count preference1:dodgeball) (count preference3:beachball) )
        3
      )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-242) (:domain medium-objects-room-v1)
(:setup
  (and
    (not
      (forall (?n - dodgeball ?f - hexagonal_bin ?l - hexagonal_bin)
        (and
          (exists (?k - dodgeball)
            (and
              (exists (?s - doggie_bed ?m - ball ?a - dodgeball)
                (game-conserved
                  (in_motion ?a ?l)
                )
              )
            )
          )
        )
      )
    )
    (game-conserved
      (agent_holds agent ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - hexagonal_bin ?i - ball ?u - shelf ?q - triangular_ramp)
        (at-end
          (not
            (in_motion rug ?q)
          )
        )
      )
    )
    (forall (?j - doggie_bed ?v ?e ?m - building)
      (and
        (preference preference2
          (then
            (hold (and (exists (?n - teddy_bear ?r - (either block dodgeball doggie_bed)) (not (in_motion ?r) ) ) (in_motion left) ) )
            (once (game_over ?v ?m) )
            (once (not (in_motion ?v ?v) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (and
      (< (count preference2:yellow) (+ (count-once-per-objects preference1:hexagonal_bin) (count-once preference1:blue) (total-score) )
      )
    )
    (>= (* (count preference2:blue_pyramid_block:basketball) (= (total-score) 5 )
        8
      )
      (count preference2:pyramid_block:beachball)
    )
  )
)
(:scoring
  (count-total preference1:triangle_block:pyramid_block)
)
)


(define (game game-id-243) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (and
        (in_motion ?xxx ?xxx)
        (equal_z_position ?xxx)
      )
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - cube_block)
        (exists (?k - dodgeball ?n - (either alarm_clock top_drawer))
          (exists (?d - dodgeball)
            (then
              (once (in_motion ?d ?d) )
              (once (on ?d) )
              (hold-for 2 (agent_holds ?d ?d) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-same-positions preference1:beachball) (count-measure preference1:dodgeball) )
    (< 100 10 )
  )
)
(:scoring
  10
)
)


(define (game game-id-244) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?j - (either triangle_block book))
      (and
        (exists (?q ?w ?u - red_dodgeball)
          (and
            (game-optional
              (in_motion ?u ?q ?w)
            )
          )
        )
        (game-conserved
          (not
            (in_motion ?j)
          )
        )
        (game-conserved
          (is_setup_object ?j ?j)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?y - ball)
      (and
        (preference preference1
          (then
            (hold (same_color bed ?y) )
            (once (agent_holds door pink_dodgeball) )
            (hold (not (in_motion ?y rug) ) )
          )
        )
        (preference preference2
          (exists (?o - block ?h - dodgeball)
            (exists (?t - pyramid_block)
              (then
                (once (and (agent_holds ?y) (in_motion pink) ) )
                (once (agent_holds agent ?h) )
                (once (and (< (distance 6 side_table) (distance ?t ?t)) ) )
              )
            )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (exists (?m - ball)
          (same_type ?m)
        )
      )
    )
    (forall (?x - tan_cube_block)
      (and
        (preference preference4
          (then
            (hold (in_motion ?x bed) )
            (once (agent_holds ?x) )
            (hold (in ?x ?x) )
            (hold (not (rug_color_under pink bed) ) )
          )
        )
        (preference preference5
          (then
            (hold (and (agent_holds ?x) (and (or (and (not (in_motion north_wall) ) (and (and (on ?x) (< (distance ?x room_center) (distance ?x green_golfball)) ) (not (in_motion ?x ?x) ) ) ) (and (in_motion ?x) (same_color ?x ?x) (and (in_motion ?x) (not (not (same_type ?x) ) ) ) (agent_holds ?x) ) ) (and (not (exists (?t - cube_block) (touch agent agent) ) ) (agent_holds ?x) ) ) ) )
            (hold-to-end (and (and (in_motion ?x) (<= 0.5 1) ) (in ?x) ) )
            (hold (agent_holds ?x) )
          )
        )
      )
    )
    (preference preference6
      (exists (?k - (either triangle_block lamp alarm_clock tall_cylindrical_block cd laptop tall_cylindrical_block) ?n - (either dodgeball cd golfball))
        (exists (?x - cube_block)
          (at-end
            (not
              (not
                (and
                  (not
                    (not
                      (in_motion ?x ?x)
                    )
                  )
                  (adjacent ?n)
                )
              )
            )
          )
        )
      )
    )
    (preference preference7
      (exists (?o - (either yellow_cube_block book))
        (exists (?t - teddy_bear)
          (then
            (hold (and (same_type ?o ?t) (agent_holds ?o) ) )
            (once (or (agent_holds ?o ?o) (agent_holds ?o) (in_motion door) (agent_holds ?o ?t) ) )
            (once (agent_holds ?t) )
          )
        )
      )
    )
    (preference preference8
      (then
        (once (in_motion ?xxx bridge_block agent) )
        (hold (on front_left_corner ?xxx) )
        (once (= 6 (distance bed ?xxx)) )
      )
    )
  )
)
(:terminal
  (< 1 (count preference8:basketball) )
)
(:scoring
  (count-once-per-objects preference1:basketball:yellow)
)
)


(define (game game-id-245) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (adjacent ?xxx) ) )
      )
    )
    (preference preference2
      (exists (?x - game_object)
        (exists (?c - desk_shelf)
          (exists (?b - chair ?e - pyramid_block)
            (exists (?v - curved_wooden_ramp)
              (exists (?n - dodgeball)
                (at-end
                  (adjacent ?c agent)
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
  (>= (count-once-per-objects preference1:hexagonal_bin) 3 )
)
(:scoring
  (count preference2:yellow)
)
)


(define (game game-id-246) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (> (distance 1 ?xxx) (distance 9 room_center))
      )
      (or
        (exists (?x - dodgeball ?a - bridge_block ?v - rug)
          (game-conserved
            (agent_holds ?v ?v)
          )
        )
        (exists (?k - (either top_drawer tall_cylindrical_block cube_block))
          (game-conserved
            (and
              (or
                (and
                  (and
                    (< 3 (distance ))
                    (on ?k)
                  )
                  (is_setup_object ?k ?k)
                )
                (in_motion ?k ?k)
              )
              (in ?k)
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
        (once (not (not (agent_holds ?xxx) ) ) )
        (once (in_motion ?xxx) )
        (once (on ?xxx) )
      )
    )
  )
)
(:terminal
  (and
    (or
      (and
        (>= (count-once-per-objects preference1:doggie_bed) (+ (* (count-once-per-objects preference1:beachball:blue_cube_block) (* (= (* (count preference1:yellow:red:golfball) (count preference1:golfball) )
                  (+ (count preference1:cube_block) (external-forall-maximize (count preference1:beachball) ) )
                )
                3
              )
              (* 4 (/ 5 (count-increasing-measure preference1:yellow_cube_block) ) )
              (* (total-score) (count preference1:yellow_cube_block) (- (count-once-per-objects preference1:dodgeball) )
                5
                (* 5 (* (external-forall-minimize (* (* (* (count-once-per-objects preference1:pyramid_block) (+ (count-once-per-objects preference1:green) (count preference1:dodgeball:basketball) )
                          )
                          (count-once preference1:blue_dodgeball:doggie_bed)
                        )
                        (count preference1:blue_dodgeball:doggie_bed)
                      )
                    )
                    5
                  )
                )
              )
              (count-overlapping preference1:dodgeball)
              (count-measure preference1)
            )
            (* 4 (+ (* (count-same-positions preference1:beachball) (count-once-per-external-objects preference1:red) )
                2
              )
            )
          )
        )
        (not
          (>= (total-time) (external-forall-maximize (count-once-per-objects preference1:blue_cube_block) ) )
        )
        (>= (count-once preference1:green) (count preference1:hexagonal_bin) )
      )
      (> (< 5 (- 5 )
        )
        1
      )
      (= 4 3 )
    )
  )
)
(:scoring
  (* (* 4 (+ (* (- (count preference1) (not (+ (* (count-once preference1:pink) (count preference1:dodgeball) )
                (total-score)
              )
            )
          )
          (count-once preference1:dodgeball)
          3
        )
        (count preference1:golfball:pink_dodgeball:green)
      )
    )
    (count-once preference1:beachball)
  )
)
)


(define (game game-id-247) (:domain many-objects-room-v1)
(:setup
  (exists (?x - game_object)
    (and
      (game-conserved
        (and
          (not
            (in_motion ?x ?x)
          )
          (agent_holds ?x ?x)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?u - wall)
      (and
        (preference preference1
          (then
            (once (not (rug_color_under ?u) ) )
            (hold (agent_holds ?u ?u) )
            (once (not (on ?u ?u) ) )
          )
        )
        (preference preference2
          (exists (?x - game_object)
            (exists (?o - (either pyramid_block yellow))
              (exists (?f ?a - hexagonal_bin)
                (then
                  (hold (or (in_motion ?a) (not (and (in ?a) (on ?u ?o) ) ) ) )
                  (hold (game_start ?x rug) )
                  (once (in ?f ?a) )
                )
              )
            )
          )
        )
        (preference preference3
          (exists (?n - dodgeball)
            (then
              (once (in_motion bed ?u) )
              (hold-while (in_motion desk) (agent_holds ?u) )
              (hold (and (in_motion ?n) (in_motion bed) (and (on ?n ?n) (and (in ?u) (not (on ?n) ) ) ) (and (and (and (not (on ?n ?u) ) (on ?n) (not (in_motion ?u ?n) ) ) (not (and (on ?u) (and (not (and (not (not (and (touch north_wall ?n) (and (not (in_motion ?n bed) ) (and (in_motion ?u) (and (in_motion ?u) (agent_holds ?u) ) ) ) ) ) ) (in ?n ?n) ) ) (agent_holds ?n ?n) ) (forall (?o - dodgeball ?b - block ?f - book) (not (adjacent desk ?u) ) ) (not (on ?u agent) ) (not (in ?u) ) (and (and (in ?u) (>= 1 (distance room_center ?n)) ) (in pink bed) (same_color agent) ) (not (in_motion ?u) ) ) ) ) (and (agent_holds front) (in ?u) ) ) (and (not (and (in_motion ?n) (agent_holds ?n ?n) ) ) (and (agent_holds ?u ?u) (not (and (not (opposite ?u floor) ) (and (agent_holds agent) (in_motion ?u ?n) ) (agent_holds ?n ?n) ) ) (< (distance ?n) (distance ?n)) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:hexagonal_bin:pink_dodgeball) (* (* (+ 5 5 )
        8
      )
      (* 10 (count preference3:blue_dodgeball) )
    )
  )
)
(:scoring
  18
)
)


(define (game game-id-248) (:domain medium-objects-room-v1)
(:setup
  (exists (?s - beachball ?x - dodgeball)
    (game-conserved
      (and
        (and
          (in_motion ?x)
          (and
            (not
              (and
                (on rug ?x)
                (and
                  (or
                    (touch ?x)
                    (not
                      (on ?x)
                    )
                  )
                  (adjacent ?x)
                )
              )
            )
            (and
              (touch ?x ?x)
              (in_motion ?x)
            )
          )
        )
        (on desk)
        (and
          (not
            (agent_holds tan)
          )
          (in_motion ?x ?x)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (not (and (and (agent_holds ?xxx) (agent_holds ?xxx bed) ) (and (and (and (in_motion ?xxx) (in_motion ?xxx ?xxx) ) (on ?xxx blue) (not (on agent) ) ) (and (not (and (in ?xxx ?xxx) (in_motion ?xxx rug) ) ) (and (and (not (and (not (on bed green) ) ) ) (in_motion ?xxx bed) ) (toggled_on ?xxx) (agent_holds ?xxx bed) (agent_holds ?xxx) ) ) ) ) ) )
        (hold (in ?xxx desk) )
        (once (adjacent ?xxx) )
      )
    )
    (preference preference2
      (then
        (hold (and (not (and (not (not (touch ?xxx) ) ) (touch ?xxx door) ) ) (and (agent_holds ?xxx) (not (in pink ?xxx) ) ) ) )
        (hold-while (agent_crouches bed) (agent_holds ?xxx) (and (and (not (and (agent_holds ?xxx) (agent_holds ?xxx) ) ) (in ?xxx rug) ) (and (and (forall (?z - ball) (in_motion ?z ?z) ) (and (in_motion ?xxx ?xxx) (>= (distance room_center ?xxx) (distance desk ?xxx)) ) ) (agent_holds ?xxx) ) ) )
        (once (in_motion ?xxx agent) )
      )
    )
  )
)
(:terminal
  (>= 3 (> 4 0.5 )
  )
)
(:scoring
  100
)
)


(define (game game-id-249) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (agent_holds ?xxx)
    )
    (exists (?x - cube_block)
      (exists (?e - cube_block)
        (and
          (exists (?k - hexagonal_bin ?k - hexagonal_bin)
            (forall (?w - triangular_ramp ?o - building)
              (game-conserved
                (or
                  (in_motion ?e ?e)
                  (and
                    (and
                      (not
                        (or
                          (same_color ?x)
                          (agent_holds ?e)
                          (on rug)
                        )
                      )
                      (and
                        (in_motion agent)
                        (agent_holds top_shelf)
                        (not
                          (not
                            (or
                              (and
                                (on ?k ?k)
                                (and
                                  (not
                                    (not
                                      (< 0 (distance room_center 6))
                                    )
                                  )
                                  (= (distance room_center ?o) (distance ?e ?x) (distance ?o agent))
                                  (and
                                    (agent_holds ?k)
                                    (in ?o)
                                    (in_motion ?k ?x)
                                  )
                                  (in_motion bed ?o)
                                )
                              )
                            )
                          )
                        )
                        (agent_holds ?o)
                      )
                    )
                    (agent_holds ?k)
                    (and
                      (on agent ?e)
                      (and
                        (not
                          (in_motion )
                        )
                        (and
                          (agent_holds ?o)
                          (in_motion ?x)
                          (exists (?l - hexagonal_bin ?z ?m - block ?q - hexagonal_bin)
                            (not
                              (and
                                (in_motion ?q ?q)
                                (= (distance ?o ?x) (distance 2 ?q))
                                (in ?x ?e)
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
          (exists (?n - dodgeball)
            (forall (?d - curved_wooden_ramp ?s - bridge_block)
              (forall (?a - ball)
                (and
                  (exists (?t - hexagonal_bin)
                    (exists (?v - wall)
                      (forall (?j - hexagonal_bin)
                        (exists (?m ?i - hexagonal_bin)
                          (game-conserved
                            (agent_holds ?t ?n)
                          )
                        )
                      )
                    )
                  )
                  (forall (?r - ball ?i - building)
                    (game-conserved
                      (not
                        (in_motion ?s)
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
      (agent_holds bed ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (and (not (agent_holds ?xxx) ) ) (not (and (and (not (< (distance ?xxx room_center) 7) ) (not (adjacent ?xxx ?xxx) ) ) (< (distance_side desk agent) 7) ) ) (and (on pink) (in_motion ?xxx ?xxx) ) )
        (once (in agent ?xxx) )
        (once (on floor) )
      )
    )
    (preference preference2
      (at-end
        (and
          (and
            (not
              (in_motion bed top_shelf)
            )
            (forall (?h - (either dodgeball bed pyramid_block))
              (in_motion ?h)
            )
            (adjacent ?xxx)
            (and
              (in_motion ?xxx ?xxx)
              (on ?xxx ?xxx)
              (in_motion ?xxx)
              (in_motion ?xxx ?xxx)
              (in ?xxx)
              (exists (?h - pillow)
                (agent_holds ?h agent)
              )
            )
          )
          (and
            (= 2 (distance room_center ?xxx))
            (object_orientation ?xxx)
          )
          (agent_holds ?xxx)
        )
      )
    )
  )
)
(:terminal
  (>= (* 4 (count-once preference1:pink:blue_cube_block) )
    (* 4 (- (count-once-per-external-objects preference2:orange) )
      (count preference1:basketball)
    )
  )
)
(:scoring
  (count preference1:pink:hexagonal_bin)
)
)


(define (game game-id-250) (:domain many-objects-room-v1)
(:setup
  (forall (?l - block)
    (game-conserved
      (in_motion ?l)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - block)
        (exists (?x - hexagonal_bin)
          (then
            (once (agent_holds ?d ?x) )
          )
        )
      )
    )
    (preference preference2
      (then
        (hold (agent_holds ?xxx ?xxx) )
        (once (agent_holds ?xxx) )
        (once (same_color ?xxx) )
        (once (and (in_motion ?xxx) (same_color ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (<= 5 (count preference1:pink:book) )
)
(:scoring
  (total-time)
)
)


(define (game game-id-251) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?e - (either dodgeball cd desktop) ?q - dodgeball)
      (and
        (and
          (forall (?n - (either dodgeball golfball golfball) ?y - blue_pyramid_block)
            (exists (?w - dodgeball)
              (exists (?i - (either blue_cube_block golfball game_object rug tall_cylindrical_block mug dodgeball))
                (game-optional
                  (and
                    (forall (?f - (either pyramid_block blue_cube_block))
                      (not
                        (in_motion ?f)
                      )
                    )
                    (on ?q ?w)
                  )
                )
              )
            )
          )
          (game-conserved
            (object_orientation ?q)
          )
        )
        (and
          (and
            (game-optional
              (and
                (agent_holds ?q)
                (and
                  (not
                    (on ?q)
                  )
                  (and
                    (> 7 1)
                    (and
                      (agent_holds agent)
                      (not
                        (and
                          (not
                            (agent_holds ?q ?q)
                          )
                          (and
                            (and
                              (on ?q ?q)
                              (agent_holds ?q ?q)
                            )
                            (touch ?q ?q)
                            (adjacent ?q)
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
    (preference preference1
      (exists (?o - dodgeball)
        (exists (?h - wall)
          (exists (?v - curved_wooden_ramp)
            (then
              (hold (not (agent_holds ?h) ) )
              (once (on ?h) )
              (once (in_motion ?h) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (+ (* 100 (count-once preference1:dodgeball) )
        (+ (* (count-once preference1:doggie_bed) (< (+ 3 2 (* (* (+ 0.7 (count preference1:pink_dodgeball) )
                    (count-once preference1:dodgeball:red:dodgeball)
                  )
                  (count-once-per-objects preference1:beachball)
                )
                0.7
                (count-once preference1:wall:yellow)
                (count preference1)
              )
              (total-score)
            )
          )
          (count preference1:yellow:dodgeball)
          (count-unique-positions preference1:green)
        )
        (* (+ (or (* (* (< (count-once-per-objects preference1:beachball:blue_dodgeball:blue_dodgeball) (- (* (count-shortest preference1:dodgeball:pink) (count preference1:blue_dodgeball) 4 )
                    )
                  )
                  (count-once preference1:golfball:pink_dodgeball)
                )
                (count-once-per-objects preference1:orange)
              )
              (> (* (/ 5 (* (total-time) (count preference1:blue_dodgeball:pink) )
                  )
                  (+ (* (+ 30 (count preference1:basketball) )
                      (count-once-per-objects preference1:blue:cube_block)
                    )
                    2
                  )
                )
                2
              )
              (* (count preference1:dodgeball) (count preference1:pink) )
            )
            (* (count-once-per-objects preference1:green) (= (not (count preference1:green) ) 2 )
            )
            (total-score)
          )
        )
        (count-same-positions preference1:white)
        (count preference1:green:dodgeball)
        (* (total-score) (* (count-once-per-objects preference1:doggie_bed) (+ (count-once-per-objects preference1:dodgeball:rug) (count preference1:beachball) )
          )
        )
      )
      3
    )
    (or
      (>= (count preference1:beachball) (- 10 )
      )
    )
    (or
      (>= (count preference1:red) 1 )
      (>= 4 (count preference1:dodgeball) )
    )
  )
)
(:scoring
  (count preference1:basketball)
)
)


(define (game game-id-252) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (not
          (on ?xxx)
        )
      )
      (game-conserved
        (in desk)
      )
      (and
        (forall (?m - dodgeball ?a ?y ?b - hexagonal_bin)
          (forall (?u - hexagonal_bin)
            (game-optional
              (touch agent)
            )
          )
        )
        (game-optional
          (adjacent_side ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (and (forall (?y - cube_block ?o - dodgeball) (adjacent_side ?o ?o) ) (on ?xxx) ) )
        (once (and (game_start ?xxx) (agent_holds ?xxx) ) )
        (hold (agent_holds agent) )
      )
    )
    (preference preference2
      (at-end
        (and
          (in_motion ?xxx pink_dodgeball)
          (in_motion ?xxx)
          (agent_holds ?xxx)
          (adjacent ?xxx)
        )
      )
    )
  )
)
(:terminal
  (>= 2 3 )
)
(:scoring
  2
)
)


(define (game game-id-253) (:domain many-objects-room-v1)
(:setup
  (forall (?b - chair)
    (game-optional
      (< 9 7)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - doggie_bed)
        (exists (?t - block)
          (exists (?o - book)
            (then
              (once (and (object_orientation ?o) (not (< 1 0.5) ) ) )
              (once (and (and (on ?r) (not (agent_holds floor) ) ) (in_motion ?t ?t) (adjacent_side ?o ?t) ) )
              (hold (agent_holds ?t ?o ?o) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ 2 1 )
    9
  )
)
(:scoring
  2
)
)


(define (game game-id-254) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (= 4 (building_size bed ?xxx))
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (in_motion ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (>= 3 (external-forall-maximize (count-once-per-objects preference1) ) )
    (> (+ 5 5 )
      (count-once-per-objects preference1:top_drawer)
    )
  )
)
(:scoring
  (+ 2 (count-once-per-objects preference1:blue_pyramid_block) )
)
)


(define (game game-id-255) (:domain medium-objects-room-v1)
(:setup
  (exists (?l - block)
    (forall (?e - hexagonal_bin)
      (and
        (and
          (and
            (game-conserved
              (and
                (> 3 1)
                (and
                  (agent_holds ?l agent)
                  (on bed)
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
    (forall (?l - game_object)
      (and
        (preference preference1
          (exists (?a - hexagonal_bin)
            (at-end
              (and
                (and
                  (not
                    (object_orientation ?l)
                  )
                  (in_motion ?l)
                  (not
                    (agent_holds bed)
                  )
                  (agent_holds ?l)
                  (agent_holds ?l)
                  (in_motion ?a ?a)
                )
                (same_type ?a ?l)
              )
            )
          )
        )
        (preference preference2
          (exists (?m - ball ?x - (either yellow_cube_block triangle_block))
            (then
              (once (and (and (in_motion ?x floor) (< 1 (distance room_center room_center)) ) (and (not (and (in_motion ?x) (in ?l ?x) ) ) (in bed) ) (in_motion ?l brown) ) )
              (hold (on ?l) )
              (once (and (not (and (rug_color_under floor ?x) (and (not (agent_holds ?x) ) (on ?l desk) (forall (?u - dodgeball) (adjacent ?l rug) ) (in_motion ?l) ) ) ) (not (in_motion ?x ?x) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* (count preference2:pink) 2 (* 2 (and 180 5 ) )
      )
      2
    )
    5
  )
)
(:scoring
  (count preference2:yellow_cube_block)
)
)

