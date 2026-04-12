import customtkinter as ctk

class LoginWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Login")
        self.geometry("200x200")
        btn = ctk.CTkButton(self, text="Login", command=self.do_login)
        btn.pack(pady=50)
        self.after(500, self.do_login)
    
    def do_login(self):
        # By calling deiconify before destroy, Tkinter never hits 0 windows
        self.master.deiconify()
        self.destroy()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Main App")
        self.geometry("400x400")
        
        self.withdraw()
        
        login = LoginWindow(self)
        self.wait_window(login)
        
        print("Init complete.")
        self.after(100, lambda: print("Main loop is alive!"))
        self.after(500, self.destroy)

if __name__ == "__main__":
    app = App()
    app.mainloop()
