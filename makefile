build:
	${MAKE} data
	sh scripts/generate.sh

start-online: 
	node dist/server/online.js 2> debug/server-online.err.log

start-debug: 
	node dist/server/online.js

start-eval:
	node dist/eval/worker.js

clean:
	@echo "Cleaning up..."
	rm -f *.prof
	@echo "Cleaned up all .prof files."

data:
	npm run build
	node dist/download.js
	npx prettier -w src --config .prettierrc