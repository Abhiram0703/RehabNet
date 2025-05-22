from tkinter import *
from tkinter.ttk import Combobox, Separator, Style, Notebook, Treeview, Frame as TFrame
from tkinter import messagebox, font
import utils
import box_and_block
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os

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
    "success": "#27ae60"     # New success green
}

def createUI():
    def start_box_and_block():
        try:
            # Get values from UI
            camera_id = cb_camera.get()
            hand = cb_hand.get()
            tolerance = scale_tolerance.get()
            username = entry_username.get().strip()
            
            if not username:
                messagebox.showerror("Error", "Please enter a username!")
                return
                
            # Validate camera
            if camera_id == "No cameras available!":
                messagebox.showerror("Error", "No cameras detected!")
                return
                
            # Convert hand to single letter
            hand = 'R' if hand == "Right" else 'L'
            
            # Get camera index (handles both "Camera 0" format and raw numbers)
            try:
                camera_index = int(camera_id.split()[-1])
            except:
                camera_index = int(camera_id)
            
            # Close UI and start game
            window.destroy()
            box_and_block.start(camera_index, hand, tolerance, username)
            
            # After game ends, show analysis dashboard
            show_analysis_dashboard(username)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {str(e)}")

    def show_analysis_dashboard(username, session_timestamp=None):
        # Load user details to find the latest session
        try:
            details_df = pd.read_csv('BBT_details.csv')
            user_sessions = details_df[details_df['Username'] == username]
            if user_sessions.empty:
                messagebox.showerror("Error", f"No session data found for user: {username}")
                return
                
            # If no specific session timestamp is provided, use the latest session
            if session_timestamp is None:
                latest_session = user_sessions.sort_values('Timestamp', ascending=False).iloc[0]
                session_timestamp = pd.to_datetime(latest_session['Timestamp']).strftime('%Y%m%d_%H%M%S')
                current_session = latest_session
            else:
                # Find the specific session
                session_datetime = datetime.strptime(session_timestamp, '%Y%m%d_%H%M%S')
                current_session = user_sessions[pd.to_datetime(user_sessions['Timestamp']).dt.strftime('%Y%m%d_%H%M%S') == session_timestamp].iloc[0]
            
            log_filename = f"./BBT_logs/BBT_logs_{username}_{session_timestamp}.csv"
            
            if not os.path.exists(log_filename):
                messagebox.showerror("Error", f"Session log file not found: {log_filename}")
                return
                
            # Load session log data
            log_df = pd.read_csv(log_filename)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load session data: {str(e)}")
            return

        # Create analysis window
        analysis_window = Tk()
        analysis_window.title(f"Box & Block Test Analysis - {username}")
        analysis_window.geometry("1200x800")
        analysis_window.configure(bg=COLORS["light"])
        
        # Custom styles for analysis window
        style = Style()
        style.theme_use('default')
        style.configure('TNotebook', background=COLORS["light"])
        style.configure('TNotebook.Tab', background=COLORS["light"], padding=[10, 5], font=('Helvetica', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', COLORS["primary"])], foreground=[('selected', COLORS["white"])])
        style.configure('Treeview', background=COLORS["white"], font=('Helvetica', 10))
        style.configure('Treeview.Heading', font=('Helvetica', 10, 'bold'))
        
        # Create notebook for tabs
        notebook = Notebook(analysis_window)
        notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        #==================================================
        # Summary Tab
        #==================================================
        summary_frame = TFrame(notebook)
        notebook.add(summary_frame, text="Summary")

        # Create a canvas and scrollbar for the summary tab
        canvas = Canvas(summary_frame, bg=COLORS["light"])
        scrollbar = Scrollbar(summary_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg=COLORS["light"])

        # Bind configure event to update scroll region
        def update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", update_scroll_region)

        # Create window in canvas, stretching to full width
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Ensure canvas resizes with window
        def update_canvas_width(event=None):
            canvas.itemconfig(canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"), 
                            width=canvas.winfo_width())
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", update_canvas_width)

        # Get summary statistics
        total_blocks = log_df['Block Number'].max()
        successful_blocks = current_session['Score']
        success_rate = (successful_blocks / total_blocks * 100) if total_blocks > 0 else 0
        avg_speed = log_df[log_df['Movement Speed'] > 0]['Movement Speed'].mean() if not log_df[log_df['Movement Speed'] > 0].empty else 0

        # Header with user profile
        header_frame = TFrame(scrollable_frame)
        header_frame.pack(fill=X, padx=20, pady=10)

        # User profile card
        profile_frame = TFrame(header_frame, style='Card.TFrame')
        profile_frame.pack(fill=X, pady=10)

        # Header with test title
        title_label = Label(profile_frame, text=f"{username}'s Box & Block Test Results", 
                        font=('Helvetica', 18, 'bold'), bg=COLORS["primary"], fg=COLORS["white"])
        title_label.pack(fill=X, ipady=10)

        # Date and hand information
        date_hand_frame = TFrame(profile_frame)
        date_hand_frame.pack(fill=X, pady=5)

        date_label = Label(date_hand_frame, text=f"Date: {current_session['Timestamp']}", 
                        font=('Helvetica', 11), bg=COLORS["light"], fg=COLORS["text"])
        date_label.pack(side=LEFT, padx=15)

        hand_label = Label(date_hand_frame, text=f"Hand Used: {current_session['Hand']}", 
                        font=('Helvetica', 11), bg=COLORS["light"], fg=COLORS["text"])
        hand_label.pack(side=LEFT, padx=15)

        # Summary statistics section
        stats_frame = TFrame(scrollable_frame)
        stats_frame.pack(fill=X, padx=20, pady=10)

        # Create stat cards in a grid layout
        stat_cards = [
            {"title": "Total Score", "value": str(int(successful_blocks)), "color": COLORS["primary"]},
            {"title": "Success Rate", "value": f"{success_rate:.1f}%", "color": COLORS["secondary"]},
            {"title": "Avg. Speed", "value": f"{avg_speed:.1f} px/s", "color": COLORS["accent"]}, 
            {"title": "Emotion Score", "value": str(current_session['Emotion Score']), "color": COLORS["warning"]},
            {"title": "Total Attempts", "value": str(int(total_blocks)), "color": COLORS["success"]}
        ]

        card_frame = Frame(stats_frame, bg=COLORS["light"])
        card_frame.pack(fill=X, pady=10)

        # Create a 3x2 grid of stat cards (5 cards, last cell empty)
        for idx, stat in enumerate(stat_cards):
            row = idx // 2
            col = idx % 2
            
            # Card container
            card = Frame(card_frame, bg=COLORS["white"], 
                    relief=RIDGE, bd=1)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Color bar at top
            color_bar = Frame(card, bg=stat["color"], height=5)
            color_bar.pack(fill=X)
            
            # Stat value (large)
            value_label = Label(card, text=stat["value"], 
                            font=('Helvetica', 24, 'bold'), 
                            bg=COLORS["white"], fg=COLORS["text"])
            value_label.pack(pady=(15, 5))
            
            # Stat title
            title_label = Label(card, text=stat["title"], 
                            font=('Helvetica', 12), 
                            bg=COLORS["white"], fg=COLORS["text"])
            title_label.pack(pady=(0, 15))

        # Configure grid
        card_frame.grid_columnconfigure(0, weight=1)
        card_frame.grid_columnconfigure(1, weight=1)
        card_frame.grid_rowconfigure(0, weight=1)
        card_frame.grid_rowconfigure(1, weight=1)
        card_frame.grid_rowconfigure(2, weight=1)

        # Emotion information
        emotion_frame = Frame(scrollable_frame, bg=COLORS["white"], relief=RIDGE, bd=1)
        emotion_frame.pack(fill=X, padx=20, pady=10)

        emotion_title = Label(emotion_frame, text="Emotional Response (based on emotion score)", 
                            font=('Helvetica', 14, 'bold'), 
                            bg=COLORS["white"], fg=COLORS["text"])
        emotion_title.pack(pady=10)

        dominant_emotion = Label(emotion_frame, 
                            text=f"Dominant Emotion: {current_session['Dominant Emotion']}", 
                            font=('Helvetica', 12), 
                            bg=COLORS["white"], fg=COLORS["text"])
        dominant_emotion.pack(pady=5)

        # Separator
        Separator(scrollable_frame, orient='horizontal').pack(fill=X, padx=20, pady=15)

        # Previous sessions section
        Label(scrollable_frame, text="Previous Sessions", 
            font=('Helvetica', 14, 'bold'), 
            bg=COLORS["light"], fg=COLORS["text"]).pack(pady=10)

        # Create frame for treeview and scrollbar
        prev_sessions_frame = TFrame(scrollable_frame)
        prev_sessions_frame.pack(fill=X, padx=20, pady=10)

        # Create treeview for previous sessions
        columns = ("Date & Time", "Hand", "Score", "Emotion Score", "Dominant Emotion")
        tree = Treeview(prev_sessions_frame, columns=columns, show="headings", selectmode="browse", height=5)

        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=CENTER)

        # Add vertical scrollbar
        scrollbar_tree = Scrollbar(prev_sessions_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar_tree.set)
        scrollbar_tree.pack(side=RIGHT, fill=Y)
        tree.pack(side=LEFT, fill=X, expand=True)

        # Add data to treeview
        sorted_sessions = user_sessions.sort_values('Timestamp', ascending=False)
        for _, session in sorted_sessions.iterrows():
            session_datetime = pd.to_datetime(session['Timestamp'])
            session_ts = session_datetime.strftime('%Y%m%d_%H%M%S')
            
            # Format the timestamp for display
            display_timestamp = session_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add session to treeview with timestamp as hidden first value
            tree.insert("", "end", values=(display_timestamp, 
                                        session['Hand'], 
                                        session['Score'], 
                                        session['Emotion Score'], 
                                        session['Dominant Emotion']),
                    tags=(session_ts,))

        # Highlight the current session
        for item in tree.get_children():
            if tree.item(item, "tags")[0] == session_timestamp:
                tree.selection_set(item)
                tree.focus(item)
                break

        # Create button frame
        button_frame = Frame(scrollable_frame, bg=COLORS["light"])
        button_frame.pack(fill=X, padx=20, pady=15)

        # View selected session button
        def view_selected_session():
            selected_item = tree.selection()
            if selected_item:
                selected_session_ts = tree.item(selected_item[0], "tags")[0]
                if selected_session_ts != session_timestamp:  # Don't reload if it's the current session
                    analysis_window.destroy()
                    show_analysis_dashboard(username, selected_session_ts)
            else:
                messagebox.showinfo("Info", "Please select a session to view")

        view_button = Button(button_frame, text="View Test Analysis", 
                        command=view_selected_session,
                        bg=COLORS["primary"], fg=COLORS["dark"], 
                        font=('Helvetica', 11, 'bold'),
                        padx=15, pady=5, relief=RAISED, bd=2)
        view_button.pack(side=LEFT, padx=5)

        # Back to main menu button
        def back_to_main():
            analysis_window.destroy()
            createUI()

        back_button = Button(button_frame, text="New session", 
                        command=back_to_main,
                        bg=COLORS["danger"], fg=COLORS["dark"], 
                        font=('Helvetica', 11, 'bold'),
                        padx=15, pady=5, relief=RAISED, bd=2)
        back_button.pack(side=LEFT, padx=5)

        # Update canvas scroll region after all widgets are added
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        #==================================================
        # Emotion Analysis Tab
        #==================================================
        emotion_frame = TFrame(notebook)
        notebook.add(emotion_frame, text="Emotion Analysis")

        if 'Score' in log_df.columns and 'Valence' in log_df.columns and 'Event Type' in log_df.columns:
            # Constants for weighting (same as calculate_emotion_score)
            MOVEMENT_EMOTION_WEIGHT = 0.7
            DROP_EMOTION_WEIGHT = 0.3
            
            # Calculate emotion score for each successful block (Score)
            emotion_by_score = []
            unique_scores = log_df[log_df['Score'] > 0]['Score'].unique()  # Only consider successful moves
            
            for score in unique_scores:
                score_data = log_df[log_df['Score'] == score]
                
                # Calculate average valence for movement and drop phases
                movement_data = score_data[score_data['Event Type'] == 'MOVEMENT']
                drop_data = score_data[score_data['Event Type'] == 'DROP']
                
                avg_movement = movement_data['Valence'].mean() if not movement_data.empty else 0
                avg_drop = drop_data['Valence'].mean() if not drop_data.empty else 0
                
                # Calculate weighted valence score
                weighted_valence = (avg_movement * MOVEMENT_EMOTION_WEIGHT) + (avg_drop * DROP_EMOTION_WEIGHT)
                
                # Determine emotion category based on valence
                if weighted_valence > 0.33:
                    emotion = "Positive"
                elif weighted_valence < -0.33:
                    emotion = "Negative"
                else:
                    emotion = "Neutral"
                    
                emotion_by_score.append(emotion)
            
            # Convert to Series for plotting
            emotion_series = pd.Series(emotion_by_score, index=unique_scores)
            
            # Create histogram
            fig1 = plt.Figure(figsize=(10, 5), dpi=100)
            ax1 = fig1.add_subplot(111)
            emotion_series.value_counts().plot(kind='bar', ax=ax1, color=COLORS["primary"])
            ax1.set_title('Emotion Distribution by Successful Block Score', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Emotion', fontsize=12)
            ax1.set_ylabel('Number of Successful Blocks', fontsize=12)
            
            # Add value labels on top of bars
            for i in ax1.patches:
                ax1.text(i.get_x() + i.get_width()/2, i.get_height() + 0.1, 
                        int(i.get_height()), ha='center', va='bottom')
            
            fig1.tight_layout()
            canvas1 = FigureCanvasTkAgg(fig1, emotion_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        else:
            # Display a message if data is not available
            message_frame = Frame(emotion_frame, bg=COLORS["white"])
            message_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
            
            Label(message_frame, text="Required data (Score, Valence, or Event Type) not available",
                font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)
            
        #==================================================
        # Movement Analysis Tab
        #==================================================
        speed_frame = TFrame(notebook)
        notebook.add(speed_frame, text="Movement Analysis")

        if 'Score' in log_df.columns and 'Movement Speed' in log_df.columns:
            # Filter for successful moves (Score > 0) and calculate average movement speed
            speed_by_score = log_df[(log_df['Score'] > 0) & (log_df['Movement Speed'] > 0)].groupby('Score')['Movement Speed'].mean()
            
            fig2 = plt.Figure(figsize=(10, 5), dpi=100)
            ax2 = fig2.add_subplot(111)
            speed_by_score.plot(kind='line', marker='o', ax=ax2, color=COLORS["secondary"], 
                                linewidth=2, markersize=8)
            ax2.set_title('Average Movement Speed per Successful Block', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Successful Block Number (Score)', fontsize=12)
            ax2.set_ylabel('Speed (px/s)', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Set x-axis to integer-only scale
            if not speed_by_score.empty:
                ax2.set_xticks(range(int(speed_by_score.index.min()), int(speed_by_score.index.max()) + 1))
            
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, speed_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        else:
            # Display a message if data is not available
            message_frame = Frame(speed_frame, bg=COLORS["white"])
            message_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
            
            Label(message_frame, text="Required data (Score or Movement Speed) not available",
                font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)
        
        #==================================================
        # Time Taken per Block Tab
        #==================================================
        time_per_block_frame = TFrame(notebook)
        notebook.add(time_per_block_frame, text="Time Analysis")
        
        if 'Score' in log_df.columns and 'Timestamp' in log_df.columns:
            # Ensure timestamp is datetime type
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            
            # First, sort the dataframe by timestamp to ensure correct order
            log_df = log_df.sort_values('Timestamp')
            
            # Find rows where Score increases (successful block moves)
            log_df['Score_Change'] = log_df['Score'].diff()
            successful_moves = log_df[log_df['Score_Change'] > 0].copy()
            
            # Calculate time differences between successful moves
            time_per_block = []
            block_numbers = []
            
            if len(successful_moves) > 0:
                # For the first successful block, calculate time since the start of session
                start_time = log_df['Timestamp'].min()
                first_success_time = successful_moves['Timestamp'].iloc[0]
                first_success_score = successful_moves['Score'].iloc[0]
                time_diff_first = (first_success_time - start_time).total_seconds()
                
                # Add first block time
                time_per_block.append(time_diff_first)
                block_numbers.append(first_success_score)
                
                # For each subsequent successful move, calculate time since previous success
                if len(successful_moves) > 1:
                    successful_moves['Time_Diff'] = successful_moves['Timestamp'].diff().dt.total_seconds()
                    
                    # Skip the first row (which has NaN diff) and use data from second row onward
                    for i in range(1, len(successful_moves)):
                        time_diff = successful_moves['Time_Diff'].iloc[i]
                        score = successful_moves['Score'].iloc[i]
                        
                        # Filter out unreasonable values
                        if pd.notna(time_diff) and 0 < time_diff < 60:  # Limit to 60 seconds per block
                            time_per_block.append(time_diff)
                            block_numbers.append(score)
            
            # Create the figure
            fig3 = plt.Figure(figsize=(10, 5), dpi=100)
            ax3 = fig3.add_subplot(111)
            
            if time_per_block:
                ax3.bar(block_numbers, time_per_block, color=COLORS["accent"])
                ax3.set_title('Time Taken per Successful Block', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Successful Block Number', fontsize=12)
                ax3.set_ylabel('Time Taken (seconds)', fontsize=12)
                
                # Format the axis
                ax3.set_ylim(0, max(time_per_block) * 1.2)  # Add 20% padding at top
                ax3.set_xticks(block_numbers)
                ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Add data labels
                for i, v in enumerate(time_per_block):
                    ax3.text(block_numbers[i], v + 0.1, f"{v:.1f}s", 
                             ha='center', va='bottom', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No successful blocks recorded', 
                        horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
                ax3.set_xlabel('Successful Block Number')
                ax3.set_ylabel('Time Taken (seconds)')
            
            fig3.tight_layout()
            canvas3 = FigureCanvasTkAgg(fig3, time_per_block_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        else:
            # Handle case where required columns aren't present
            fig3 = plt.Figure(figsize=(10, 5), dpi=100)
            ax3 = fig3.add_subplot(111)
            ax3.text(0.5, 0.5, 'Required data columns not found in log file', 
                    horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_xlabel('Successful Block Number')
            ax3.set_ylabel('Time Taken (seconds)')
            
            fig3.tight_layout()
            canvas3 = FigureCanvasTkAgg(fig3, time_per_block_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        
        #==================================================
        # Emotion vs Time Tab
        #==================================================
        emotion_time_frame = TFrame(notebook)
        notebook.add(emotion_time_frame, text="Emotion Timeline")

        if 'Timestamp' in log_df.columns:
            # Convert timestamp and extract seconds
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            log_df['Time Elapsed'] = (log_df['Timestamp'] - log_df['Timestamp'].min()).dt.total_seconds()
            
            # Create 1-second bins and get mode emotion for each bin
            log_df['Time Bin'] = log_df['Time Elapsed'].astype(int)  # Bin by whole seconds
            emotion_by_second = log_df.groupby('Time Bin')['Emotion'].agg(lambda x: x.mode()[0] if not x.empty else 'Neutral')
            
            # Convert emotion to numeric for plotting
            emotion_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            emotion_by_second_numeric = emotion_by_second.map(emotion_mapping)
            
            # Create plot
            fig4 = plt.Figure(figsize=(10, 5), dpi=100)
            ax4 = fig4.add_subplot(111)
            
            # Plot as stepped line to show constant emotion for each second
            emotion_by_second_numeric.plot(kind='line', drawstyle='steps-post', ax=ax4,
                                        color=COLORS["primary"], linewidth=2)
            
            ax4.set_title('Emotion Variation Over Time (Mode per Second)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Time Elapsed (seconds)', fontsize=12)
            ax4.set_ylabel('Emotion State', fontsize=12)
            ax4.set_yticks([-1, 0, 1])
            ax4.set_yticklabels(['Negative', 'Neutral', 'Positive'])
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Set x-axis to show integer seconds only
            if not emotion_by_second_numeric.empty:
                ax4.set_xticks(range(int(emotion_by_second_numeric.index.min()), 
                                int(emotion_by_second_numeric.index.max()) + 1))
            
            fig4.tight_layout()
            canvas4 = FigureCanvasTkAgg(fig4, emotion_time_frame)
            canvas4.draw()
            canvas4.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        
        #==================================================
        # Emotion Score Analysis Tab
        #==================================================
        emotion_block_frame = TFrame(notebook)
        notebook.add(emotion_block_frame, text="Emotion Score Analysis")

        if 'Score' in log_df.columns and 'Valence' in log_df.columns and 'Event Type' in log_df.columns:
            # Constants for weighting (as per your calculation function)
            MOVEMENT_EMOTION_WEIGHT = 0.7
            DROP_EMOTION_WEIGHT = 0.3
            
            # Process each successful block (Score) to calculate emotion score based on valence
            score_emotion_scores = []
            unique_scores = log_df[log_df['Score'] > 0]['Score'].unique()  # Only consider successful moves
            
            for score in unique_scores:
                score_data = log_df[log_df['Score'] == score]
                
                # Calculate average valence for movement and drop phases
                movement_data = score_data[score_data['Event Type'] == 'MOVEMENT']
                drop_data = score_data[score_data['Event Type'] == 'DROP']
                
                avg_movement = movement_data['Valence'].mean() if not movement_data.empty else 0
                avg_drop = drop_data['Valence'].mean() if not drop_data.empty else 0
                
                # Calculate weighted score
                weighted_valence = (avg_movement * MOVEMENT_EMOTION_WEIGHT) + (avg_drop * DROP_EMOTION_WEIGHT)
                
                # Store the raw valence score
                score_emotion_scores.append((score, weighted_valence))
            
            # Convert to Series for plotting
            emotion_score_by_score = pd.Series([score for _, score in score_emotion_scores], 
                                            index=[score for score, _ in score_emotion_scores])
            
            fig5 = plt.Figure(figsize=(10, 5), dpi=100)
            ax5 = fig5.add_subplot(111)
            
            # Ensure we have points at every integer score to avoid gaps
            if not emotion_score_by_score.empty:
                all_scores = range(1, int(max(emotion_score_by_score.index)) + 1)
                complete_series = pd.Series([emotion_score_by_score.get(score, 0) for score in all_scores], index=all_scores)
                
                # Plot using the complete series
                complete_series.plot(kind='line', marker='o', ax=ax5, 
                                color=COLORS["accent"], linewidth=2, markersize=8)
                
                # Add a horizontal line at y=0 for reference (neutral valence)
                ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                
                # Fill areas above and below the neutral line
                ax5.fill_between(complete_series.index, 
                                [max(0, val) for val in complete_series.values], 0,
                                color=COLORS["secondary"], alpha=0.3)
                ax5.fill_between(complete_series.index, 
                                [min(0, val) for val in complete_series.values], 0,
                                color=COLORS["danger"], alpha=0.3)
                
                ax5.set_title('Emotion Valence by Successful Block Number', fontsize=14, fontweight='bold')
                ax5.set_xlabel('Successful Block Number (Score)', fontsize=12)
                ax5.set_ylabel('Weighted Valence Score', fontsize=12)
                
                # Set custom y-ticks based on valence interpretation
                min_val = min(complete_series.values)
                max_val = max(complete_series.values)
                range_val = max(abs(min_val), abs(max_val))
                
                # Set appropriate y-limits with some padding
                y_limit = max(0.5, range_val * 1.2)
                ax5.set_ylim(-y_limit, y_limit)
                
                # Custom y-ticks for valence interpretation
                if range_val > 0.3:
                    ax5.set_yticks([-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5])
                    ax5.set_yticklabels(['Negative', 'Moderately Negative', 'Slightly Negative', 
                                        'Neutral', 'Slightly Positive', 'Moderately Positive', 'Positive'])
                else:
                    ax5.set_yticks([-0.3, -0.1, 0, 0.1, 0.3])
                    ax5.set_yticklabels(['Moderately Negative', 'Slightly Negative', 
                                        'Neutral', 'Slightly Positive', 'Moderately Positive'])
                
                ax5.grid(True, linestyle='--', alpha=0.7)
                
                # Set x-axis to integer-only scale
                ax5.set_xticks(range(int(min(complete_series.index)), int(max(complete_series.index)) + 1))
                
                # Add light horizontal lines at key thresholds
                ax5.axhline(y=0.1, color='lightgreen', linestyle=':', alpha=0.5)
                ax5.axhline(y=-0.1, color='lightcoral', linestyle=':', alpha=0.5)
                
                fig5.tight_layout()
                canvas5 = FigureCanvasTkAgg(fig5, emotion_block_frame)
                canvas5.draw()
                canvas5.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
            else:
                # Handle empty data case
                message_frame = Frame(emotion_block_frame, bg=COLORS["white"])
                message_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
                
                Label(message_frame, text="Emotion score data not available",
                    font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)
        else:
            # Display a message if data is not available
            message_frame = Frame(emotion_block_frame, bg=COLORS["white"])
            message_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
            
            Label(message_frame, text="Required data (Score, Valence, or Event Type) not available",
                font=('Helvetica', 14), bg=COLORS["white"], fg=COLORS["text"]).pack(pady=50)
        #==================================================
        # NEW: Trends Tab with Dropdown
        #==================================================
        trends_frame = TFrame(notebook)
        notebook.add(trends_frame, text="Trends")
        
        # Create control panel for trends
        control_frame = Frame(trends_frame, bg=COLORS["light"])
        control_frame.pack(fill=X, padx=10, pady=10)
        
        # Add dropdown for selecting trend to display
        Label(control_frame, text="Select Trend Analysis:", 
             font=('Helvetica', 12, 'bold'), 
             bg=COLORS["light"], fg=COLORS["text"]).pack(side=LEFT, padx=10)
        
        trend_options = [
            "Total Blocks Moved Over Sessions",
            "Average Movement Speed Over Sessions",
            "Emotion Score Over Sessions",
            "Time Taken Per Block Over Sessions"
        ]
        
        trend_var = StringVar()
        trend_dropdown = Combobox(control_frame, textvariable=trend_var, 
                                values=trend_options, state="readonly", width=40)
        trend_dropdown.current(0)
        trend_dropdown.pack(side=LEFT, padx=10)
        
        # Create frame for the chart
        chart_frame = Frame(trends_frame, bg=COLORS["white"])
        chart_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Function to update chart based on selection
        def update_trend_chart(*args):
            # Clear previous chart
            for widget in chart_frame.winfo_children():
                widget.destroy()
                
            selected_trend = trend_var.get()
            
            # Sort sessions by timestamp
            sorted_sessions = user_sessions.sort_values('Timestamp')
            
            # Format timestamps for x-axis
            timestamps = pd.to_datetime(sorted_sessions['Timestamp']).dt.strftime('%m/%d %H:%M')
            
            fig = plt.Figure(figsize=(10, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            if selected_trend == "Total Blocks Moved Over Sessions":
                ax.plot(timestamps, sorted_sessions['Score'], marker='o', 
                      linewidth=2, markersize=8, color=COLORS["primary"])
                ax.set_title('Total Blocks Moved Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Number of Blocks', fontsize=12)
                
            elif selected_trend == "Average Movement Speed Over Sessions":
                # Calculate average movement speed for each session
                avg_speeds = []
                for _, session in sorted_sessions.iterrows():
                    session_timestamp = pd.to_datetime(session['Timestamp']).strftime('%Y%m%d_%H%M%S')
                    log_file = f"./BBT_logs/BBT_logs_{username}_{session_timestamp}.csv"
                    
                    try:
                        if os.path.exists(log_file):
                            session_log = pd.read_csv(log_file)
                            avg_speed = session_log[session_log['Movement Speed'] > 0]['Movement Speed'].mean()
                            avg_speeds.append(avg_speed if not pd.isna(avg_speed) else 0)
                        else:
                            avg_speeds.append(0)
                    except:
                        avg_speeds.append(0)
                
                ax.plot(timestamps, avg_speeds, marker='o', 
                      linewidth=2, markersize=8, color=COLORS["secondary"])
                ax.set_title('Average Movement Speed Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Speed (px/s)', fontsize=12)
                
            elif selected_trend == "Emotion Score Over Sessions":
                ax.plot(timestamps, sorted_sessions['Emotion Score'], marker='o', 
                      linewidth=2, markersize=8, color=COLORS["accent"])
                ax.set_title('Emotion Score Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Emotion Score', fontsize=12)
                
            elif selected_trend == "Time Taken Per Block Over Sessions":
                # Calculate average time per block for each session
                avg_times = []
                for _, session in sorted_sessions.iterrows():
                    session_timestamp = pd.to_datetime(session['Timestamp']).strftime('%Y%m%d_%H%M%S')
                    log_file = f"./BBT_logs/BBT_logs_{username}_{session_timestamp}.csv"
                    
                    try:
                        if os.path.exists(log_file):
                            session_log = pd.read_csv(log_file)
                            
                            # Calculate time per block using timestamp differences
                            if 'Timestamp' in session_log.columns and 'Score' in session_log.columns:
                                session_log['Timestamp'] = pd.to_datetime(session_log['Timestamp'])
                                session_log = session_log.sort_values('Timestamp')
                                
                                # Calculate time differences for successful moves
                                session_log['Score_Change'] = session_log['Score'].diff()
                                successful_moves = session_log[session_log['Score_Change'] > 0]
                                
                                if len(successful_moves) > 1:
                                    successful_moves['Time_Diff'] = successful_moves['Timestamp'].diff().dt.total_seconds()
                                    avg_time = successful_moves['Time_Diff'].iloc[1:].mean()  # Skip first (NaN)
                                    avg_times.append(avg_time if not pd.isna(avg_time) else 0)
                                else:
                                    avg_times.append(0)
                            else:
                                avg_times.append(0)
                        else:
                            avg_times.append(0)
                    except:
                        avg_times.append(0)
                
                ax.plot(timestamps, avg_times, marker='o', 
                      linewidth=2, markersize=8, color=COLORS["warning"])
                ax.set_title('Average Time per Block Over Sessions', fontsize=14, fontweight='bold')
                ax.set_ylabel('Time (seconds)', fontsize=12)
            
            ax.set_xlabel('Session Date & Time', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            fig.tight_layout()
            
            # Add the chart to the frame
            canvas = FigureCanvasTkAgg(fig, chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Bind dropdown changes to chart update function
        trend_var.trace("w", update_trend_chart)
        
        # Initialize with the first trend
        update_trend_chart()
        
        analysis_window.mainloop()

    # Create main window
    window = Tk()
    window.title("Box & Block Test")
    window.geometry("800x1000")
    window.resizable(True, True)
    window.configure(bg=COLORS["light"])
    
    # Custom font definitions
    title_font = font.Font(family="Helvetica", size=22, weight="bold")
    header_font = font.Font(family="Helvetica", size=14, weight="bold")
    label_font = font.Font(family="Helvetica", size=12)
    button_font = font.Font(family="Helvetica", size=12, weight="bold")
    
    # Create a header frame
    header_frame = Frame(window, bg=COLORS["primary"], height=80)
    header_frame.pack(fill=X)
    
    # Application logo/title
    title_label = Label(header_frame, text="Box & Block Test", 
                      font=title_font, bg=COLORS["primary"], fg=COLORS["white"])
    title_label.pack(pady=20)
    
    # Main content container with rounded corners effect
    main_container = Frame(window, bg=COLORS["white"], bd=2, relief=RIDGE)
    main_container.pack(fill=BOTH, expand=True, padx=20, pady=20)
    
    # Add a descriptive subtitle
    subtitle = Label(main_container, 
                   text="A computerized assessment of manual dexterity",
                   font=("Helvetica", 11, "italic"), bg=COLORS["white"], fg=COLORS["text"])
    subtitle.pack(pady=(20, 30))
    
    # User information section
    user_frame = LabelFrame(main_container, text="User Information", 
                          font=header_font, bg=COLORS["white"], fg=COLORS["text"],
                          padx=15, pady=15)
    user_frame.pack(fill=X, padx=20, pady=10)
    
    # Username entry
    Label(user_frame, text="Username:", font=label_font, 
         bg=COLORS["white"], fg=COLORS["text"], anchor=W).pack(fill=X)
    entry_username = Entry(user_frame, font=label_font, bd=2, relief=SOLID)
    entry_username.pack(fill=X, pady=(5, 15))
    
    # Test configuration section
    config_frame = LabelFrame(main_container, text="Test Configuration", 
                            font=header_font, bg=COLORS["white"], fg=COLORS["text"],
                            padx=15, pady=15)
    config_frame.pack(fill=X, padx=20, pady=20)
    
    # Camera selection
    Label(config_frame, text="Select Camera:", font=label_font, 
         bg=COLORS["white"], fg=COLORS["text"], anchor=W).pack(fill=X)
    
    camera_indexes = utils.returnCameraIndexes()
    if not camera_indexes:
        camera_indexes = ["No cameras available!"]
    
    camera_frame = Frame(config_frame, bg=COLORS["white"])
    camera_frame.pack(fill=X, pady=(5, 15))
    
    camera_icon = Label(camera_frame, text="📹", font=("Helvetica", 16), 
                       bg=COLORS["white"], fg=COLORS["text"])
    camera_icon.pack(side=LEFT, padx=(0, 10))
    
    cb_camera = Combobox(camera_frame, values=camera_indexes, state="readonly", 
                       font=label_font, width=30)
    cb_camera.current(0)
    cb_camera.pack(side=LEFT, fill=X, expand=True)
    
    # Hand selection
    Label(config_frame, text="Select Hand:", font=label_font, 
         bg=COLORS["white"], fg=COLORS["text"], anchor=W).pack(fill=X)
    
    hand_frame = Frame(config_frame, bg=COLORS["white"])
    hand_frame.pack(fill=X, pady=(5, 15))
    
    hand_icon = Label(hand_frame, text="👐", font=("Helvetica", 16), 
                     bg=COLORS["white"], fg=COLORS["text"])
    hand_icon.pack(side=LEFT, padx=(0, 10))
    
    cb_hand = Combobox(hand_frame, values=("Left", "Right"), state="readonly", 
                     font=label_font, width=30)
    cb_hand.current(1)  # Default to Right
    cb_hand.pack(side=LEFT, fill=X, expand=True)
    
    # Tolerance slider with improved visual
    Label(config_frame, text="Exercise Tolerance:", font=label_font, 
         bg=COLORS["white"], fg=COLORS["text"], anchor=W).pack(fill=X)
    
    # Create frame for slider and value display
    slider_frame = Frame(config_frame, bg=COLORS["white"])
    slider_frame.pack(fill=X, pady=(5, 5))
    
    # Icon for tolerance
    tolerance_icon = Label(slider_frame, text="⚙️", font=("Helvetica", 16), 
                         bg=COLORS["white"], fg=COLORS["text"])
    tolerance_icon.pack(side=LEFT, padx=(0, 10))
    
    # Slider container
    slider_container = Frame(slider_frame, bg=COLORS["white"])
    slider_container.pack(side=LEFT, fill=X, expand=True)
    
    scale_tolerance = Scale(slider_container, from_=1.0, to=4.5, resolution=0.1, 
                          orient=HORIZONTAL, length=250, 
                          sliderrelief=RAISED, bd=1,
                          highlightthickness=0,
                          bg=COLORS["white"],
                          troughcolor=COLORS["light"],
                          activebackground=COLORS["primary"])
    scale_tolerance.set(2.5)  # Default value
    scale_tolerance.pack(fill=X)
    
    # Current tolerance value display
    tolerance_value = Label(config_frame, text=f"Current Value: {scale_tolerance.get()}", 
                          font=("Helvetica", 10, "italic"), 
                          bg=COLORS["white"], fg=COLORS["text"])
    tolerance_value.pack(fill=X, pady=(0, 10))
    
    # Update value label when slider moves
    scale_tolerance.config(command=lambda v: tolerance_value.config(text=f"Current Value: {float(v):.1f}"))
    
    # Start button with better visibility
    button_frame = Frame(main_container, bg=COLORS["white"], pady=20, padx=20)
    button_frame.pack(fill=X, expand=True)

    btn_start = Button(button_frame, text="START TEST", command=start_box_and_block,
                    bg=COLORS["success"], fg=COLORS["dark"], 
                    font=button_font,
                    width=20, height=2,
                    relief=RAISED, bd=3,
                    activebackground=COLORS["primary"], 
                    activeforeground=COLORS["white"])
    btn_start.pack(pady=20, ipadx=10, ipady=5)

    # Information box
    info_frame = Frame(main_container, bg=COLORS["light"], bd=1, relief=SOLID, padx=15, pady=15)
    info_frame.pack(fill=X, padx=20, pady=10)
    
    info_text = """The Box and Block Test measures unilateral gross manual dexterity. 
The test requires moving as many blocks as possible from one compartment to another within a time limit.
The camera will track your hand movement during the exercise."""
    
    info_label = Label(info_frame, text=info_text, bg=COLORS["light"], 
                     font=("Helvetica", 10), fg=COLORS["text"], justify=LEFT, wraplength=480)
    info_label.pack(fill=X)
    
    # Disable start if no cameras
    if camera_indexes[0] == "No cameras available!":
        btn_start.config(state=DISABLED, bg=COLORS["light"], fg=COLORS["text"])
        
    # Credits at bottom
    credits = Label(window, text="© 2025 Box & Block Test System", 
                  font=("Helvetica", 8), bg=COLORS["light"], fg=COLORS["text"])
    credits.pack(side=BOTTOM, pady=5)
    
    window.mainloop()

if __name__ == "__main__":
    createUI()