import tkinter as tk
from tkinter import ttk, messagebox, font
from tkinter import Canvas, Frame, Scrollbar
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os
import numpy as np
import csv
import letter_writing

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

def returnCameraIndexes():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(f"Camera {index}")
        cap.release()
        index += 1
    return arr if arr else ["No cameras available!"]

def createUI():
    def start_letter_writing():
        try:
            camera_id = cb_camera.get()
            hand = cb_hand.get()
            username = entry_username.get().strip()
            
            if not username:
                messagebox.showerror("Error", "Please enter a username!")
                return
            if ',' in username:
                messagebox.showerror("Error", "Username cannot contain commas!")
                return
            if camera_id == "No cameras available!":
                messagebox.showerror("Error", "No cameras detected!")
                return
                
            hand = 'R' if hand == "Right" else 'L'
            try:
                camera_index = int(camera_id.split()[-1])
            except:
                camera_index = int(camera_id)
            
            window.destroy()
            letter_writing.start(camera_index, hand, username)
            show_analysis_dashboard(username)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {str(e)}")

    def show_analysis_dashboard(username, session_timestamp=None):
        try:
            details_df = pd.read_csv('letter_writing_details.csv', on_bad_lines='warn', quoting=csv.QUOTE_MINIMAL)
            user_sessions = details_df[details_df['Username'] == username]
            if user_sessions.empty:
                messagebox.showerror("Error", f"No session data found for user: {username}")
                return
                
            if session_timestamp is None:
                latest_session = user_sessions.sort_values('Timestamp', ascending=False).iloc[0]
                session_timestamp = pd.to_datetime(latest_session['Timestamp']).strftime('%Y%m%d_%H%M%S')
                current_session = latest_session
            else:
                session_datetime = datetime.strptime(session_timestamp, '%Y%m%d_%H%M%S')
                current_session = user_sessions[pd.to_datetime(user_sessions['Timestamp']).dt.strftime('%Y%m%d_%H%M%S') == session_timestamp].iloc[0]
            
            log_filename = f"./LW_logs/letter_writing_logs_{username}_{session_timestamp}.csv"
            
            if not os.path.exists(log_filename):
                messagebox.showerror("Error", f"Session log file not found: {log_filename}")
                return
                
            log_df = pd.read_csv(log_filename, on_bad_lines='warn', quoting=csv.QUOTE_MINIMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load session data: {str(e)}")
            return

        analysis_window = tk.Tk()
        analysis_window.title(f"Letter Writing Analysis - {username}")
        analysis_window.geometry("1200x800")
        analysis_window.configure(bg=COLORS["light"])
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=COLORS["light"])
        style.configure('TNotebook.Tab', background=COLORS["light"], padding=[10, 5], font=('Helvetica', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', COLORS["primary"])], foreground=[('selected', COLORS["white"])])
        style.configure('Treeview', background=COLORS["white"], font=('Helvetica', 10))
        style.configure('Treeview.Heading', font=('Helvetica', 10, 'bold'))
        style.configure('Card.TFrame', background=COLORS["white"], relief=tk.RIDGE, borderwidth=1)

        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary Tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")

        canvas = tk.Canvas(summary_frame, bg=COLORS["light"])
        scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS["light"])

        def update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", update_scroll_region)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def update_canvas_width(event=None):
            canvas.itemconfig(canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"), 
                            width=canvas.winfo_width())
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", update_canvas_width)

        total_letters = len(log_df['Letter'].unique())
        completion_time = current_session['Completion Time']
        avg_speed = current_session['Avg Movement Speed']
        avg_error = current_session['Avg Error Distance']

        header_frame = tk.Frame(scrollable_frame, bg=COLORS["light"])
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        profile_frame = ttk.Frame(header_frame, style='Card.TFrame')
        profile_frame.pack(fill=tk.X, pady=10)

        title_label = tk.Label(profile_frame, text=f"{username}'s Letter Writing Results", 
                            font=('Helvetica', 18, 'bold'), bg=COLORS["primary"], fg=COLORS["white"])
        title_label.pack(fill=tk.X, ipady=10)

        date_hand_frame = tk.Frame(profile_frame, bg=COLORS["light"])
        date_hand_frame.pack(fill=tk.X, pady=5)

        date_label = tk.Label(date_hand_frame, text=f"Date: {current_session['Timestamp']}", 
                            font=('Helvetica', 11), bg=COLORS["light"], fg=COLORS["text"])
        date_label.pack(side=tk.LEFT, padx=15)

        hand_label = tk.Label(date_hand_frame, text=f"Hand Used: {current_session['Hand']}", 
                            font=('Helvetica', 11), bg=COLORS["light"], fg=COLORS["text"])
        hand_label.pack(side=tk.LEFT, padx=15)

        stats_frame = tk.Frame(scrollable_frame, bg=COLORS["light"])
        stats_frame.pack(fill=tk.X, padx=20, pady=10)

        stat_cards = [
            {"title": "Total Letters", "value": str(total_letters), "color": COLORS["primary"]},
            {"title": "Completion Time", "value": f"{completion_time:.1f}s", "color": COLORS["secondary"]},
            {"title": "Avg Speed", "value": f"{avg_speed:.1f} px/s", "color": COLORS["accent"]},
            {"title": "Avg Error", "value": f"{avg_error:.1f} mm", "color": COLORS["warning"]},
            {"title": "Dominant Emotion", "value": current_session['Dominant Emotion'], "color": COLORS["success"]}
        ]

        card_frame = tk.Frame(stats_frame, bg=COLORS["light"])
        card_frame.pack(fill=tk.X, pady=10)

        for idx, stat in enumerate(stat_cards):
            row = idx // 2
            col = idx % 2
            card = tk.Frame(card_frame, bg=COLORS["white"], relief=tk.RIDGE, bd=1)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            color_bar = tk.Frame(card, bg=stat["color"], height=5)
            color_bar.pack(fill=tk.X)
            value_label = tk.Label(card, text=stat["value"], font=('Helvetica', 24, 'bold'), 
                                bg=COLORS["white"], fg=COLORS["text"])
            value_label.pack(pady=(15, 5))
            title_label = tk.Label(card, text=stat["title"], font=('Helvetica', 12), 
                                bg=COLORS["white"], fg=COLORS["text"])
            title_label.pack(pady=(0, 15))

        card_frame.grid_columnconfigure(0, weight=1)
        card_frame.grid_columnconfigure(1, weight=1)
        card_frame.grid_rowconfigure(0, weight=1)
        card_frame.grid_rowconfigure(1, weight=1)
        card_frame.grid_rowconfigure(2, weight=1)

        emotion_frame = tk.Frame(scrollable_frame, bg=COLORS["white"], relief=tk.RIDGE, bd=1)
        emotion_frame.pack(fill=tk.X, padx=20, pady=10)

        emotion_title = tk.Label(emotion_frame, text="Emotional Response", 
                                font=('Helvetica', 14, 'bold'), bg=COLORS["white"], fg=COLORS["text"])
        emotion_title.pack(pady=10)

        dominant_emotion = tk.Label(emotion_frame, text=f"Dominant Emotion: {current_session['Dominant Emotion']}", 
                                font=('Helvetica', 12), bg=COLORS["white"], fg=COLORS["text"])
        dominant_emotion.pack(pady=5)

        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, padx=20, pady=15)

        tk.Label(scrollable_frame, text="Previous Sessions", font=('Helvetica', 14, 'bold'), 
                bg=COLORS["light"], fg=COLORS["text"]).pack(pady=10)

        prev_sessions_frame = ttk.Frame(scrollable_frame)
        prev_sessions_frame.pack(fill=tk.X, padx=20, pady=10)

        columns = ("Date & Time", "Hand", "Completion Time", "Dominant Emotion", "Avg Speed", "Avg Error")
        tree = ttk.Treeview(prev_sessions_frame, columns=columns, show="headings", selectmode="browse", height=5)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=tk.CENTER)

        scrollbar_tree = ttk.Scrollbar(prev_sessions_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar_tree.set)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.X, expand=True)

        sorted_sessions = user_sessions.sort_values('Timestamp', ascending=False)
        for _, session in sorted_sessions.iterrows():
            session_datetime = pd.to_datetime(session['Timestamp'])
            session_ts = session_datetime.strftime('%Y%m%d_%H%M%S')
            display_timestamp = session_datetime.strftime('%Y-%m-%d %H:%M:%S')
            tree.insert("", "end", values=(display_timestamp, 
                                        session['Hand'], 
                                        f"{session['Completion Time']:.1f}s", 
                                        session['Dominant Emotion'],
                                        f"{session['Avg Movement Speed']:.1f} px/s",
                                        f"{session['Avg Error Distance']:.1f} mm"),
                        tags=(session_ts,))

        for item in tree.get_children():
            if tree.item(item, "tags")[0] == session_timestamp:
                tree.selection_set(item)
                tree.focus(item)
                break

        button_frame = tk.Frame(scrollable_frame, bg=COLORS["light"])
        button_frame.pack(fill=tk.X, padx=20, pady=15)

        def view_selected_session():
            selected_item = tree.selection()
            if selected_item:
                selected_session_ts = tree.item(selected_item[0], "tags")[0]
                if selected_session_ts != session_timestamp:
                    analysis_window.destroy()
                    show_analysis_dashboard(username, selected_session_ts)
            else:
                messagebox.showinfo("Info", "Please select a session to view")

        view_button = tk.Button(button_frame, text="View Test Analysis", command=view_selected_session,
                               bg=COLORS["primary"], fg=COLORS["dark"], font=('Helvetica', 11, 'bold'),
                               padx=15, pady=5, relief=tk.RAISED, bd=2)
        view_button.pack(side=tk.LEFT, padx=5)

        def back_to_main():
            analysis_window.destroy()
            createUI()

        back_button = tk.Button(button_frame, text="New Session", command=back_to_main,
                               bg=COLORS["danger"], fg=COLORS["dark"], font=('Helvetica', 11, 'bold'),
                               padx=15, pady=5, relief=tk.RAISED, bd=2)
        back_button.pack(side=tk.LEFT, padx=5)

        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Emotion Variation Tab with Dropdown
        emotion_var_frame = ttk.Frame(notebook)
        notebook.add(emotion_var_frame, text="Emotion Variation")

        control_frame = tk.Frame(emotion_var_frame, bg=COLORS["light"])
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Select Letter for Emotion Variation:", 
                font=('Helvetica', 12, 'bold'), bg=COLORS["light"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=10)

        letter_var = tk.StringVar()
        letter_dropdown = ttk.Combobox(control_frame, textvariable=letter_var, 
                                     values=list(log_df['Letter'].unique()), state="readonly", width=10)
        letter_dropdown.current(0)
        letter_dropdown.pack(side=tk.LEFT, padx=10)

        chart_frame_emotion = tk.Frame(emotion_var_frame, bg=COLORS["white"])
        chart_frame_emotion.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def update_emotion_chart(*args):
            for widget in chart_frame_emotion.winfo_children():
                widget.destroy()
                
            selected_letter = letter_var.get()
            letter_data = log_df[(log_df['Letter'] == selected_letter) & (log_df['Movement Speed'] > 0)]
            
            if not letter_data.empty and 'Timestamp' in letter_data.columns and 'Emotion' in letter_data.columns:
                letter_data['Timestamp'] = pd.to_datetime(letter_data['Timestamp'])
                letter_data['Time Elapsed'] = (letter_data['Timestamp'] - letter_data['Timestamp'].min()).dt.total_seconds()
                
                emotion_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
                letter_data['Emotion Numeric'] = letter_data['Emotion'].map(emotion_mapping)
                
                fig4 = plt.Figure(figsize=(10, 5), dpi=100)
                ax4 = fig4.add_subplot(111)
                ax4.plot(letter_data['Time Elapsed'], letter_data['Emotion Numeric'], drawstyle='steps-post',
                        color=COLORS["primary"], linewidth=2)
                ax4.set_title(f'Emotion Variation for Letter {selected_letter}', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Time Elapsed (seconds)', fontsize=12)
                ax4.set_ylabel('Emotion State', fontsize=12)
                ax4.set_yticks([-1, 0, 1])
                ax4.set_yticklabels(['Negative', 'Neutral', 'Positive'])
                ax4.grid(True, linestyle='--', alpha=0.7)
                
                if not letter_data.empty:
                    ax4.set_xlim(0, letter_data['Time Elapsed'].max())
                
                fig4.tight_layout()
                canvas4 = FigureCanvasTkAgg(fig4, chart_frame_emotion)
                canvas4.draw()
                canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                message_frame = tk.Frame(chart_frame_emotion, bg=COLORS["white"])
                message_frame.pack(fill=tk.BOTH, expand=True)
                tk.Label(message_frame, text="No movement or emotion data available for this letter",
                        font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)

        letter_var.trace("w", update_emotion_chart)
        update_emotion_chart()

        # Movement Analysis Tab
        speed_frame = ttk.Frame(notebook)
        notebook.add(speed_frame, text="Movement Analysis")

        if 'Letter' in log_df.columns and 'Movement Speed' in log_df.columns:
            speed_by_letter = log_df[log_df['Movement Speed'] > 0].groupby('Letter')['Movement Speed'].mean()
            
            fig2 = plt.Figure(figsize=(10, 5), dpi=100)
            ax2 = fig2.add_subplot(111)
            speed_by_letter.plot(kind='bar', ax=ax2, color=COLORS["secondary"])
            ax2.set_title('Average Movement Speed per Letter', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Letter', fontsize=12)
            ax2.set_ylabel('Speed (px/s)', fontsize=12)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            for i, v in enumerate(speed_by_letter):
                ax2.text(i, v + 0.1, f"{v:.1f}", ha='center', va='bottom')
            
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, speed_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            message_frame = tk.Frame(speed_frame, bg=COLORS["white"])
            message_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            tk.Label(message_frame, text="Required data (Letter or Movement Speed) not available",
                    font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)

        # Error Analysis Tab
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Error Analysis")

        if 'Letter' in log_df.columns and 'Avg Error Distance' in log_df.columns:
            error_by_letter = log_df.groupby('Letter')['Avg Error Distance'].mean()
            
            fig3 = plt.Figure(figsize=(10, 5), dpi=100)
            ax3 = fig3.add_subplot(111)
            error_by_letter.plot(kind='bar', ax=ax3, color=COLORS["accent"])
            ax3.set_title('Average Avg Error Distance per Letter', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Letter', fontsize=12)
            ax3.set_ylabel('Avg Error Distance (mm)', fontsize=12)
            ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            for i, v in enumerate(error_by_letter):
                ax3.text(i, v + 0.1, f"{v:.1f}", ha='center', va='bottom')
            
            fig3.tight_layout()
            canvas3 = FigureCanvasTkAgg(fig3, error_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            message_frame = tk.Frame(error_frame, bg=COLORS["white"])
            message_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            tk.Label(message_frame, text="Required data (Letter or Avg Error Distance) not available",
                    font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)

        # Emotion vs Performance Tab
        emotion_perf_frame = ttk.Frame(notebook)
        notebook.add(emotion_perf_frame, text="Emotion vs Performance")

        control_frame = tk.Frame(emotion_perf_frame, bg=COLORS["light"])
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Select Metric for Emotion Correlation:", 
                font=('Helvetica', 12, 'bold'), bg=COLORS["light"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=10)

        metric_var = tk.StringVar()
        metric_dropdown = ttk.Combobox(control_frame, textvariable=metric_var, 
                                      values=["Movement Speed", "Avg Error Distance"], state="readonly", width=20)
        metric_dropdown.current(0)
        metric_dropdown.pack(side=tk.LEFT, padx=10)

        chart_frame = tk.Frame(emotion_perf_frame, bg=COLORS["white"])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def update_emotion_perf_chart(*args):
            for widget in chart_frame.winfo_children():
                widget.destroy()
                
            selected_metric = metric_var.get()
            if selected_metric == "Movement Speed":
                metric_column = 'Movement Speed'
                ylabel = 'Speed (px/s)'
                title = 'Emotion vs Movement Speed'
            else:
                metric_column = 'Avg Error Distance'
                ylabel = 'Avg Error Distance (mm)'
                title = 'Emotion vs Avg Error Distance'
            
            if 'Letter' in log_df.columns and 'Emotion' in log_df.columns and metric_column in log_df.columns:
                grouped_data = log_df[log_df[metric_column] > 0].groupby(['Letter', 'Emotion'])[metric_column].mean().unstack()
                
                fig5 = plt.Figure(figsize=(10, 5), dpi=100)
                ax5 = fig5.add_subplot(111)
                grouped_data.plot(kind='bar', ax=ax5, color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]])
                ax5.set_title(title, fontsize=14, fontweight='bold')
                ax5.set_xlabel('Letter', fontsize=12)
                ax5.set_ylabel(ylabel, fontsize=12)
                ax5.grid(True, axis='y', linestyle='--', alpha=0.7)
                ax5.legend(title='Emotion')
                
                fig5.tight_layout()
                canvas5 = FigureCanvasTkAgg(fig5, chart_frame)
                canvas5.draw()
                canvas5.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                message_frame = tk.Frame(chart_frame, bg=COLORS["white"])
                message_frame.pack(fill=tk.BOTH, expand=True)
                tk.Label(message_frame, text=f"No data available for {selected_metric} correlation",
                        font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)

        metric_var.trace("w", update_emotion_perf_chart)
        update_emotion_perf_chart()

        # Trends Tab
        trends_frame = ttk.Frame(notebook)
        notebook.add(trends_frame, text="Trends")

        control_frame = tk.Frame(trends_frame, bg=COLORS["light"])
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Select Trend Analysis:", 
                font=('Helvetica', 12, 'bold'), bg=COLORS["light"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=10)

        trend_options = [
            "Completion Time Over Sessions",
            "Average Movement Speed Over Sessions",
            "Average Avg Error Distance Over Sessions",
            "Emotion Stability Over Sessions"
        ]
        
        trend_var = tk.StringVar()
        trend_dropdown = ttk.Combobox(control_frame, textvariable=trend_var, 
                                    values=trend_options, state="readonly", width=40)
        trend_dropdown.current(0)
        trend_dropdown.pack(side=tk.LEFT, padx=10)

        chart_frame_trends = tk.Frame(trends_frame, bg=COLORS["white"])
        chart_frame_trends.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def update_trend_chart(*args):
            for widget in chart_frame_trends.winfo_children():
                widget.destroy()
                
            selected_trend = trend_var.get()
            sorted_sessions = user_sessions.sort_values('Timestamp')
            timestamps = pd.to_datetime(sorted_sessions['Timestamp']).dt.strftime('%m/%d %H:%M')
            
            fig = plt.Figure(figsize=(10, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            if selected_trend == "Completion Time Over Sessions":
                ax.plot(timestamps, sorted_sessions['Completion Time'], marker='o', 
                      linewidth=2, markersize=8, color=COLORS["primary"])
                ax.set_title('Completion Time Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Time (seconds)', fontsize=12)
                
            elif selected_trend == "Average Movement Speed Over Sessions":
                ax.plot(timestamps, sorted_sessions['Avg Movement Speed'], marker='o', 
                      linewidth=2, markersize=8, color=COLORS["secondary"])
                ax.set_title('Average Movement Speed Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Speed (px/s)', fontsize=12)
                
            elif selected_trend == "Average Avg Error Distance Over Sessions":
                ax.plot(timestamps, sorted_sessions['Avg Error Distance'], marker='o', 
                      linewidth=2, markersize=8, color=COLORS["accent"])
                ax.set_title('Average Avg Error Distance Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Avg Error Distance (mm)', fontsize=12)
                
            elif selected_trend == "Emotion Stability Over Sessions":
                stability_scores = []
                for _, session in sorted_sessions.iterrows():
                    session_ts = pd.to_datetime(session['Timestamp']).strftime('%Y%m%d_%H%M%S')
                    log_file = f"./LW_logs/letter_writing_logs_{username}_{session_ts}.csv"
                    try:
                        if os.path.exists(log_file):
                            session_log = pd.read_csv(log_file, on_bad_lines='warn', quoting=csv.QUOTE_MINIMAL)
                            neutral_count = len(session_log[session_log['Emotion'] == 'Neutral'])
                            total_count = len(session_log)
                            stability = neutral_count / total_count if total_count > 0 else 0
                            stability_scores.append(stability)
                        else:
                            stability_scores.append(0)
                    except:
                        stability_scores.append(0)
                
                ax.plot(timestamps, stability_scores, marker='o', 
                      linewidth=2, markersize=8, color=COLORS["warning"])
                ax.set_title('Emotion Stability Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Proportion of Neutral Emotion', fontsize=12)
                ax.set_ylim(0, 1)
            
            ax.set_xlabel('Session Date & Time', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, chart_frame_trends)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        trend_var.trace("w", update_trend_chart)
        update_trend_chart()

        analysis_window.mainloop()

    window = tk.Tk()
    window.title("Letter Writing Test")
    window.geometry("800x1000")
    window.resizable(True, True)
    window.configure(bg=COLORS["light"])

    title_font = font.Font(family="Helvetica", size=22, weight="bold")
    header_font = font.Font(family="Helvetica", size=14, weight="bold")
    label_font = font.Font(family="Helvetica", size=12)
    button_font = font.Font(family="Helvetica", size=12, weight="bold")

    header_frame = tk.Frame(window, bg=COLORS["primary"], height=80)
    header_frame.pack(fill=tk.X)

    title_label = tk.Label(header_frame, text="Letter Writing Test", 
                         font=title_font, bg=COLORS["primary"], fg=COLORS["white"])
    title_label.pack(pady=20)

    main_container = tk.Frame(window, bg=COLORS["white"], bd=2, relief=tk.RIDGE)
    main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    subtitle = tk.Label(main_container, 
                       text="A computerized assessment of fine motor skills through letter tracing",
                       font=("Helvetica", 11, "italic"), bg=COLORS["white"], fg=COLORS["text"])
    subtitle.pack(pady=(20, 30))

    user_frame = tk.LabelFrame(main_container, text="User Information", 
                              font=header_font, bg=COLORS["white"], fg=COLORS["text"],
                              padx=15, pady=15)
    user_frame.pack(fill=tk.X, padx=20, pady=10)

    tk.Label(user_frame, text="Username:", font=label_font, 
            bg=COLORS["white"], fg=COLORS["text"], anchor=tk.W).pack(fill=tk.X)
    entry_username = tk.Entry(user_frame, font=label_font, bd=2, relief=tk.SOLID)
    entry_username.pack(fill=tk.X, pady=(5, 15))

    config_frame = tk.LabelFrame(main_container, text="Test Configuration", 
                                font=header_font, bg=COLORS["white"], fg=COLORS["text"],
                                padx=15, pady=15)
    config_frame.pack(fill=tk.X, padx=20, pady=20)

    tk.Label(config_frame, text="Select Camera:", font=label_font, 
            bg=COLORS["white"], fg=COLORS["text"], anchor=tk.W).pack(fill=tk.X)

    camera_indexes = returnCameraIndexes()
    camera_frame = tk.Frame(config_frame, bg=COLORS["white"])
    camera_frame.pack(fill=tk.X, pady=(5, 15))

    camera_icon = tk.Label(camera_frame, text="📹", font=("Helvetica", 16), 
                         bg=COLORS["white"], fg=COLORS["text"])
    camera_icon.pack(side=tk.LEFT, padx=(0, 10))

    cb_camera = ttk.Combobox(camera_frame, values=camera_indexes, state="readonly", 
                            font=label_font, width=30)
    cb_camera.current(0)
    cb_camera.pack(side=tk.LEFT, fill=tk.X, expand=True)

    tk.Label(config_frame, text="Select Hand:", font=label_font, 
            bg=COLORS["white"], fg=COLORS["text"], anchor=tk.W).pack(fill=tk.X)

    hand_frame = tk.Frame(config_frame, bg=COLORS["white"])
    hand_frame.pack(fill=tk.X, pady=(5, 15))

    hand_icon = tk.Label(hand_frame, text="👐", font=("Helvetica", 16), 
                        bg=COLORS["white"], fg=COLORS["text"])
    hand_icon.pack(side=tk.LEFT, padx=(0, 10))

    cb_hand = ttk.Combobox(hand_frame, values=("Left", "Right"), state="readonly", 
                          font=label_font, width=30)
    cb_hand.current(1)
    cb_hand.pack(side=tk.LEFT, fill=tk.X, expand=True)

    button_frame = tk.Frame(main_container, bg=COLORS["white"], pady=20, padx=20)
    button_frame.pack(fill=tk.X, expand=True)

    btn_start = tk.Button(button_frame, text="START TEST", command=start_letter_writing,
                         bg=COLORS["success"], fg=COLORS["dark"], font=button_font,
                         width=20, height=2, relief=tk.RAISED, bd=3,
                         activebackground=COLORS["primary"], activeforeground=COLORS["white"])
    btn_start.pack(pady=20, ipadx=10, ipady=5)

    info_frame = tk.Frame(main_container, bg=COLORS["light"], bd=1, relief=tk.SOLID, padx=15, pady=15)
    info_frame.pack(fill=tk.X, padx=20, pady=10)

    info_text = """The Letter Writing Test assesses fine motor skills and emotional response 
by tracing letters using hand tracking. The camera will monitor your hand movements 
and facial expressions during the task."""
    
    info_label = tk.Label(info_frame, text=info_text, bg=COLORS["light"], 
                         font=("Helvetica", 10), fg=COLORS["text"], justify=tk.LEFT, wraplength=480)
    info_label.pack(fill=tk.X)

    if camera_indexes[0] == "No cameras available!":
        btn_start.config(state=tk.DISABLED, bg=COLORS["light"], fg=COLORS["text"])

    credits = tk.Label(window, text="© 2025 Letter Writing Test System", 
                     font=("Helvetica", 8), bg=COLORS["light"], fg=COLORS["text"])
    credits.pack(side=tk.BOTTOM, pady=5)

    window.mainloop()

if __name__ == "__main__":
    createUI()
    