up:
	docker compose -f docker-compose.yaml up $(OPTIONS)

down:
	docker compose -f docker-compose.yaml down $(OPTIONS)

.PHONY: up down