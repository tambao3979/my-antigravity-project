import customtkinter as ctk

class LoginWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Login")
        self.geometry("200x200")
        btn = ctk.CTkButton(self, text="Login", command=self.do_login)
        btn.pack(pady=50)
    
    def do_login(self):
        self.destroy()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Main App")
        self.geometry("400x400")
        
        self.withdraw()
        
        login = LoginWindow(self)
        self.wait_window(login)
        
        self.deiconify()
        print("Main window deiconified. Init complete.")

if __name__ == "__main__":
    app = App()
    print("Calling mainloop...")
    app.mainloop()
    print("Mainloop exited!")
