default:
	@echo "Targets include: build"

build:
	docker compose build

up:
	docker compose up -d

build_up:
	docker compose up -d --build

migrate:
	docker compose run -it --rm backend alembic upgrade head
