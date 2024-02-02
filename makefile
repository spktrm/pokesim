build:
	tsc

start-online: 
	${MAKE} build
	node dist/server/online.js 2> debug/server-online.err.log

start-debug: 
	tsc
	node dist/server/online.js debug 2> debug/server-online.err.log

start-eval:
	tsc
	node dist/eval/worker.js

clean:
	@echo "Cleaning up..."
	rm -f *.prof
	@echo "Cleaned up all .prof files."

data:
	tsc
	node dist/download.js
	npx prettier -w src --config .prettierrc