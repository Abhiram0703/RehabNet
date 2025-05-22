import tkinter as tk
from tkinter import font, messagebox
import sys

try:
    import writing_letters_main
    import main_box_and_block
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Color scheme consistent with provided scripts
COLORS = {
    "primary": "#3498db",    # Blue
    "secondary": "#2ecc71",  # Green
    "accent": "#9b59b6",     # Purple
    "danger": "#e74c3c",     # Red
    "warning": "#f39c12",    # Orange
    "light": "#f5f5f5",      # Light Gray
    "dark": "#333333",       # Dark Gray
    "text": "#2c3e50",       # Dark Blue-Gray
    "white": "#ffffff",      # White
    "success": "#27ae60"     # Success Green
}

def create_main_ui():
    def start_letter_writing():
        window.destroy()
        try:
            writing_letters_main.createUI()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start Letter Writing Test: {str(e)}")
            create_main_ui()  # Restart main UI on failure

    def start_box_and_block():
        window.destroy()
        try:
            main_box_and_block.createUI()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start Box and Block Test: {str(e)}")
            create_main_ui()  # Restart main UI on failure

    # Create main window
    window = tk.Tk()
    window.title("Motor Skills Assessment")
    window.geometry("800x600")
    window.resizable(True, True)
    window.configure(bg=COLORS["light"])

    # Font definitions
    title_font = font.Font(family="Helvetica", size=24, weight="bold")
    subtitle_font = font.Font(family="Helvetica", size=12, slant="italic")
    button_font = font.Font(family="Helvetica", size=14, weight="bold")
    info_font = font.Font(family="Helvetica", size=10)

    # Header frame
    header_frame = tk.Frame(window, bg=COLORS["primary"], height=100)
    header_frame.pack(fill=tk.X)

    title_label = tk.Label(
        header_frame,
        text="Motor Skills Assessment",
        font=title_font,
        bg=COLORS["primary"],
        fg=COLORS["white"]
    )
    title_label.pack(pady=30)

    # Main content container
    main_container = tk.Frame(window, bg=COLORS["white"], bd=2, relief=tk.RIDGE)
    main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Subtitle
    subtitle = tk.Label(
        main_container,
        text="Select a test to assess fine motor skills and dexterity",
        font=subtitle_font,
        bg=COLORS["white"],
        fg=COLORS["text"]
    )
    subtitle.pack(pady=(20, 30))

    # Test selection frame
    selection_frame = tk.LabelFrame(
        main_container,
        text="Choose Test",
        font=("Helvetica", 14, "bold"),
        bg=COLORS["white"],
        fg=COLORS["text"],
        padx=15,
        pady=15
    )
    selection_frame.pack(fill=tk.X, padx=20, pady=20)

    # Button frame for test buttons
    button_frame = tk.Frame(selection_frame, bg=COLORS["white"])
    button_frame.pack(fill=tk.X, pady=20)

    # Letter Writing Test button
    btn_letter_writing = tk.Button(
        button_frame,
        text="Letter Writing Test",
        command=start_letter_writing,
        bg=COLORS["success"],
        fg=COLORS["dark"],
        font=button_font,
        width=20,
        height=2,
        relief=tk.RAISED,
        bd=3,
        activebackground=COLORS["primary"],
        activeforeground=COLORS["white"]
    )
    btn_letter_writing.pack(side=tk.LEFT, padx=20, pady=10)

    # Box and Block Test button
    btn_box_and_block = tk.Button(
        button_frame,
        text="Box and Block Test",
        command=start_box_and_block,
        bg=COLORS["secondary"],
        fg=COLORS["dark"],
        font=button_font,
        width=20,
        height=2,
        relief=tk.RAISED,
        bd=3,
        activebackground=COLORS["primary"],
        activeforeground=COLORS["white"]
    )
    btn_box_and_block.pack(side=tk.LEFT, padx=20, pady=10)

    # Information box
    info_frame = tk.Frame(main_container, bg=COLORS["light"], bd=1, relief=tk.SOLID, padx=15, pady=15)
    info_frame.pack(fill=tk.X, padx=20, pady=20)

    info_text = (
        "Choose a test to begin:\n"
        "- Letter Writing Test: Assess fine motor skills by tracing letters using hand tracking.\n"
        "- Box and Block Test: Measure manual dexterity by moving blocks between compartments."
    )
    info_label = tk.Label(
        info_frame,
        text=info_text,
        bg=COLORS["light"],
        font=info_font,
        fg=COLORS["text"],
        justify=tk.LEFT,
        wraplength=700
    )
    info_label.pack(fill=tk.X)

    # Credits
    credits = tk.Label(
        window,
        text="© 2025 Motor Skills Assessment System",
        font=("Helvetica", 8),
        bg=COLORS["light"],
        fg=COLORS["text"]
    )
    credits.pack(side=tk.BOTTOM, pady=10)

    window.mainloop()

if __name__ == "__main__":
    create_main_ui()