explain select count(*) from title t, cast_info ci where t.id=ci.movie_id and t.kind_id = 7 and t.production_year = 1992 and ci.person_id > 2415257 and ci.role_id < 3;