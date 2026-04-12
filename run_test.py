import sys
import traceback

with open("crash_log.txt", "w") as f:
    # Redirect stderr directly to file BEFORE importing gui
    sys.stderr = f
    
    try:
        import gui
        # Monkey patch LoginWindow so it authenticates instantly
        def fake_wait(self, window):
            window.authenticated = True
            window.destroy()
            
        gui.SecurityApp.wait_window = fake_wait
        
        # Don't let stdout/stderr redirect hide the crash
        class FakeRedirect:
            def __init__(self, t): pass
            def write(self, s): pass
            def flush(self): pass
        gui._StdoutRedirect = FakeRedirect
        
        app = gui.SecurityApp()
        
        print("Starting mainloop", file=f)
        f.flush()
        
        app.after(500, lambda: sys.exit(0)) # Exit successfully if it survives 500ms
        app.mainloop()
    except BaseException as e:
        print("Exception caught at top level:", file=f)
        traceback.print_exc(file=f)
