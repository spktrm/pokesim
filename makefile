clean:
	@echo "Cleaning up..."
	rm -f *.prof
	@echo "Cleaned up all .prof files."

start: 
	tsc
	node dist/main.js 2> debug/server.err.log