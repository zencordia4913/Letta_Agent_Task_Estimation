up:
	docker compose -f docker-compose.yaml up $(OPTIONS)

down:
	docker compose -f docker-compose.yaml down $(OPTIONS)

kill:
	docker compose -f docker-compose.yaml down -v

.PHONY: up down kill