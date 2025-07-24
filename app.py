import os
import cv2
import numpy as np
import uuid
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import login_required
from werkzeug.utils import secure_filename
import pdfkit

# Flask App Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dictionary to store processing tasks and their progress
processing_tasks = {}

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Database Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load User for Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

# Upload & Process Video
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    output_video_url = None  # Default to None

    if request.method == 'POST':
        # Check if it's an AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type and 'multipart/form-data' in request.content_type
        
        if 'video' not in request.files:
            if is_ajax:
                return jsonify({'status': 'error', 'message': 'No file part'})
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['video']

        if file.filename == '':
            if is_ajax:
                return jsonify({'status': 'error', 'message': 'No selected file'})
            flash('No selected file', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):  # Ensure it's a valid video format
            if is_ajax:
                return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload an MP4, AVI, or MOV file.'})
            flash('Invalid file type. Please upload an MP4, AVI, or MOV file.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = "output_video.mp4"  # Output file will be stored as MP4
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        file.save(input_path)
        
        # Generate a unique ID for this processing task
        processing_id = str(uuid.uuid4())
        
        # Initialize the processing task with 0% progress
        processing_tasks[processing_id] = {
            'progress': 0,
            'status': 'processing',
            'input_path': input_path,
            'output_path': output_path,
            'output_filename': output_filename
        }
        
        # Start processing in a background thread
        processing_thread = threading.Thread(
            target=process_video_with_progress,
            args=(processing_id, input_path, output_path)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        if is_ajax:
            return jsonify({
                'status': 'success',
                'message': 'File uploaded successfully. Processing started.',
                'processing_id': processing_id
            })
        
        flash('File uploaded successfully. Processing...', 'success')
        # For non-AJAX requests, redirect to a page that shows progress
        return redirect(url_for('upload'))

    return render_template('upload.html', output_video_url=output_video_url)


# Check Processing Progress
@app.route('/check_progress')
def check_progress():
    processing_id = request.args.get('id')
    
    if not processing_id or processing_id not in processing_tasks:
        return jsonify({'status': 'error', 'message': 'Invalid processing ID'})
    
    task = processing_tasks[processing_id]
    
    if task['status'] == 'completed':
        # Return the URL to the processed video
        redirect_url = url_for('upload') + '?video=' + task['output_filename']
        return jsonify({
            'status': 'completed',
            'progress': 100,
            'redirect_url': redirect_url
        })
    elif task['status'] == 'error':
        return jsonify({
            'status': 'error',
            'message': task.get('error_message', 'An error occurred during processing')
        })
    else:
        return jsonify({
            'status': 'processing',
            'progress': task['progress']
        })


# Cancel Processing
@app.route('/cancel_processing', methods=['POST'])
def cancel_processing():
    data = request.get_json()
    processing_id = data.get('processing_id')
    
    if not processing_id or processing_id not in processing_tasks:
        return jsonify({'status': 'error', 'message': 'Invalid processing ID'})
    
    # Mark the task as cancelled
    processing_tasks[processing_id]['status'] = 'cancelled'
    
    return jsonify({'status': 'success', 'message': 'Processing cancelled'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Allowed file extensions for video upload
def allowed_file(filename):
    allowed_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Process Video Function with Progress Tracking
def process_video_with_progress(processing_id, input_path, output_path):
    try:
        # Update task status
        processing_tasks[processing_id]['progress'] = 1
        
        # Read video frames
        video_frames = read_video(input_path)
        if not video_frames:
            processing_tasks[processing_id]['status'] = 'error'
            processing_tasks[processing_id]['error_message'] = 'No frames extracted from video.'
            return
        
        processing_tasks[processing_id]['progress'] = 10
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # Track objects
        tracker = Tracker('models/best.pt')
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
        tracker.add_position_to_tracks(tracks)
        
        processing_tasks[processing_id]['progress'] = 25
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # Camera movement estimation
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
        
        processing_tasks[processing_id]['progress'] = 40
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # View transformation
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)
        
        processing_tasks[processing_id]['progress'] = 50
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # Ball interpolation
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        
        processing_tasks[processing_id]['progress'] = 60
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # Speed and distance estimation
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
        
        processing_tasks[processing_id]['progress'] = 70
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # Team assignment
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        
        processing_tasks[processing_id]['progress'] = 80
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # Player-ball assignment
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 'None')
        
        processing_tasks[processing_id]['progress'] = 90
        if processing_tasks[processing_id]['status'] == 'cancelled':
            return
        
        # Draw annotations and save video
        team_ball_control = np.array(team_ball_control)
        # Generate a detailed HTML report after processing
        report_filename = f"report_{processing_id}.html"
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
        # --- Stable team assignment: assign team based on first frame seen ---
        player_team_map = {}
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                if player_id not in player_team_map:
                    player_team_map[player_id] = track.get('team', 'N/A')
        # --- Player details extraction with stable team ---
        player_summary = {}
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                stable_team = player_team_map.get(player_id, track.get('team', 'N/A'))
                if player_id not in player_summary:
                    player_summary[player_id] = {
                        'team': stable_team,
                        'team_color': track.get('team_color', 'N/A'),
                        'distance': 0.0,
                        'speed_sum': 0.0,
                        'speed_count': 0,
                        'frames': 0,
                        'ball_possession': 0
                    }
                # Distance (if available)
                if 'distance' in track:
                    player_summary[player_id]['distance'] += track['distance']
                # Speed (if available)
                if 'speed' in track:
                    player_summary[player_id]['speed_sum'] += track['speed']
                    player_summary[player_id]['speed_count'] += 1
                player_summary[player_id]['frames'] += 1
                # Ball possession
                if track.get('has_ball', False):
                    player_summary[player_id]['ball_possession'] += 1
        # --- Helper: Map color value to color name ---
        def color_to_name(color):
            # Accepts tuple (R,G,B) or hex string
            color_map = {
                (255, 0, 0): 'Red',
                (0, 0, 255): 'Blue',
                (0, 255, 0): 'Green',
                (255, 255, 0): 'Yellow',
                (255, 255, 255): 'White',
                (0, 0, 0): 'Black',
                (255, 165, 0): 'Orange',
                (128, 0, 128): 'Purple',
                (0, 255, 255): 'Cyan',
                (255, 192, 203): 'Pink',
                (128, 128, 128): 'Gray',
                (128, 128, 0): 'Olive',
                (0, 128, 0): 'Dark Green',
                (0, 128, 128): 'Teal',
                (128, 0, 0): 'Maroon',
                (0, 0, 128): 'Navy',
            }
            if isinstance(color, tuple) and len(color) == 3:
                return color_map.get(tuple(int(x) for x in color), str(color))
            if isinstance(color, str):
                # Try to parse hex string
                try:
                    if color.startswith('#') and len(color) == 7:
                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                        return color_map.get(rgb, color)
                except Exception:
                    pass
            return str(color)
        # --- Group and limit players per team (using stable team) ---
        from collections import defaultdict
        team_players = defaultdict(list)
        for pid, info in player_summary.items():
            team_players[info['team']].append((pid, info))
        # Limit to 11 players per team
        for team in team_players:
            team_players[team] = team_players[team][:11]
        # --- Prepare team player ID lists for Team 1 and Team 2 ---
        team1_ids = [str(pid) for pid, info in team_players.get('Team 1', [])]
        team2_ids = [str(pid) for pid, info in team_players.get('Team 2', [])]
        team_id_section = ""
        if team1_ids:
            team_id_section += f"<p><b>Team 1 Player IDs:</b> {', '.join(team1_ids)}</p>"
        if team2_ids:
            team_id_section += f"<p><b>Team 2 Player IDs:</b> {', '.join(team2_ids)}</p>"
        # --- Sort teams: Team 1, Team 2, then others ---
        def team_order_key(team):
            t = str(team).lower()
            if t == 'team 1':
                return (0, t)
            elif t == 'team 2':
                return (1, t)
            else:
                return (2, t)
        sorted_teams = sorted(team_players.keys(), key=team_order_key)
        # --- Modern, visually appealing and PDF-friendly HTML report (strictly matching requested format) ---
        report_html = f"""
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='UTF-8'>
            <meta name='viewport' content='width=device-width, initial-scale=1.0'>
            <title>Football Match Analysis Report</title>
            <style>
                body {{
                    background: #f7f9fb;
                    font-family: 'Roboto', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    color: #222;
                }}
                .header-bar {{
                    background: #0072ff;
                    color: white;
                    padding: 32px 0 24px 0;
                    text-align: center;
                    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                    position: relative;
                }}
                .header-bar .icon {{
                    display: inline-block;
                    vertical-align: middle;
                    margin-right: 12px;
                    width: 2.5rem;
                    height: 2.5rem;
                }}
                .header-bar h1 {{
                    display: inline-block;
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin: 0;
                    letter-spacing: 1px;
                    vertical-align: middle;
                }}
                .container {{
                    max-width: 900px;
                    margin: 32px auto;
                    background: #fff;
                    border-radius: 18px;
                    box-shadow: 0 4px 32px rgba(0,0,0,0.10);
                    padding: 32px 24px 32px 24px;
                }}
                .meta-card {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 24px;
                    background: #f3f7fa;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    padding: 20px 24px;
                    margin-bottom: 32px;
                    border-left: 6px solid #00c6ff;
                }}
                .meta-label {{
                    font-weight: 700;
                    margin-right: 8px;
                    color: #0072ff;
                }}
                .meta-value {{
                    font-weight: 400;
                    color: #222;
                }}
                .meta-status.completed {{
                    color: #22bb33;
                    font-weight: 700;
                    background: #e6f9ea;
                    border-radius: 6px;
                    padding: 2px 10px;
                    margin-left: 8px;
                }}
                .meta-status.error {{
                    color: #bb2222;
                    font-weight: 700;
                    background: #fbeaea;
                    border-radius: 6px;
                    padding: 2px 10px;
                    margin-left: 8px;
                }}
                .meta-timestamp {{
                    font-size: 0.98rem;
                    color: #888;
                    font-style: italic;
                    margin-top: 6px;
                }}
                .team-ids {{
                    margin-bottom: 24px;
                }}
                .team-ids span {{
                    display: inline-block;
                    background: #eaf4ff;
                    color: #0072ff;
                    border-radius: 8px;
                    padding: 4px 12px;
                    margin: 0 8px 8px 0;
                    font-weight: 500;
                    font-size: 1.05rem;
                }}
                .team-section h3 {{
                    margin-top: 32px;
                    margin-bottom: 10px;
                    font-size: 1.35rem;
                    color: #0072ff;
                    letter-spacing: 0.5px;
                }}
                .player-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 32px;
                    background: #f9fbfd;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
                }}
                .player-table th {{
                    background: #0072ff;
                    color: #fff;
                    font-weight: 700;
                    padding: 12px 8px;
                    font-size: 1.08rem;
                    border-right: 1px solid #eaf4ff;
                }}
                .player-table th:last-child {{
                    border-right: none;
                }}
                .player-table td {{
                    padding: 10px 8px;
                    border-bottom: 1px solid #eaf4ff;
                    font-size: 1.04rem;
                    text-align: center;
                }}
                .player-table tr:nth-child(even) {{
                    background: #f3f7fa;
                }}
                .player-table tr:last-child td {{
                    border-bottom: none;
                }}
                .badge-possession {{
                    display: inline-block;
                    padding: 2px 10px;
                    border-radius: 8px;
                    font-weight: 600;
                    color: #fff;
                    background: #22bb33;
                    font-size: 0.98rem;
                }}
                .badge-possession.low {{
                    background: #ff9800;
                }}
                .badge-possession.verylow {{
                    background: #bb2222;
                }}
                .color-circle {{
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    margin-right: 6px;
                    border: 2px solid #000;
                    vertical-align: middle;
                }}
                .icon-metric {{
                    font-style: normal;
                    font-weight: bold;
                    margin-right: 3px;
                }}
                @media (max-width: 700px) {{
                    .container {{ padding: 12px 2px; }}
                    .meta-card {{ flex-direction: column; gap: 10px; padding: 12px; }}
                    .player-table th, .player-table td {{ font-size: 0.95rem; padding: 7px 2px; }}
                }}
            </style>
        </head>
        <body>
            <div class='header-bar'>
                <span class='icon'>
                    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='40' height='40' fill='currentColor'>
                        <circle cx='16' cy='16' r='16' fill='#fff' opacity='0.15'/>
                        <circle cx='16' cy='16' r='13' fill='#fff' opacity='0.25'/>
                        <circle cx='16' cy='16' r='10' fill='#0072ff'/>
                        <circle cx='16' cy='16' r='6' fill='#fff'/>
                        <circle cx='16' cy='16' r='2.5' fill='#0072ff'/>
                    </svg>
                </span>
                <h1>Football Match Analysis Report</h1>
            </div>
            <div class='container'>
                <div class='meta-card'>
                    <div><span class='meta-label'>Processing ID:</span> <span class='meta-value'>{processing_id}</span></div>
                    <div><span class='meta-label'>Input Video:</span> <span class='meta-value'>{os.path.basename(input_path)}</span></div>
                    <div><span class='meta-label'>Output Video:</span> <span class='meta-value'>{os.path.basename(output_path)}</span></div>
                    <div><span class='meta-label'>Status:</span> <span class='meta-status completed'>Completed</span></div>
                </div>
                <div class='meta-timestamp'>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div class='team-ids'>
                    {('<span><b>Team 1 Player IDs:</b> ' + ', '.join(team1_ids) + '</span>') if team1_ids else ''}
                    {('<span><b>Team 2 Player IDs:</b> ' + ', '.join(team2_ids) + '</span>') if team2_ids else ''}
                </div>
        """
        for team in sorted_teams:
            report_html += f"<div class='team-section'><h3>{team}</h3>"
            report_html += """
            <table class='player-table'>
                <tr>
                    <th>Player ID</th>
                    <th>Team Color</th>
                    <th>Total Distance</th>
                    <th><span class='icon-metric' style='vertical-align:middle;'>
                        <svg width='18' height='18' viewBox='0 0 24 24' style='vertical-align:middle;'>
                            <circle cx='12' cy='12' r='10' fill='#0072ff' opacity='0.15'/>
                            <path d='M12 6v6l4 2' stroke='#0072ff' stroke-width='2' fill='none' stroke-linecap='round'/>
                            <circle cx='12' cy='12' r='2' fill='#0072ff'/>
                        </svg>
                    </span>Avg Speed</th>
                    <th>Frames Present</th>
                    <th><span class='icon-metric'>&#9917;</span>Ball Possession</th>
                </tr>
            """
            for pid, info in team_players[team]:
                avg_speed = info['speed_sum'] / info['speed_count'] if info['speed_count'] else 0.0
                color_val = info['team_color']
                if str(info['team']).lower() == 'team 1':
                    rgb_str = 'rgb(255,255,255)'
                    color_name_disp = 'White'
                elif str(info['team']).lower() == 'team 2':
                    rgb_str = 'rgb(0,255,0)'
                    color_name_disp = 'Green'
                else:
                    if isinstance(color_val, tuple) and len(color_val) == 3:
                        rgb_str = f"rgb({color_val[0]},{color_val[1]},{color_val[2]})"
                    elif isinstance(color_val, str) and color_val.startswith('#'):
                        rgb_str = color_val
                    else:
                        rgb_str = '#888'
                    color_name_disp = color_to_name(color_val)
                possession = info['ball_possession']
                if possession > 40:
                    badge_class = 'badge-possession'
                elif possession > 15:
                    badge_class = 'badge-possession low'
                else:
                    badge_class = 'badge-possession verylow'
                report_html += f"<tr>"
                report_html += f"<td>{pid}</td>"
                report_html += f"<td><span class='color-circle' style='background:{rgb_str}; border:2px solid #000;'></span>{color_name_disp}</td>"
                report_html += f"<td>{info['distance']:.2f} m</td>"
                report_html += (
                    f"<td><span class='icon-metric' style='vertical-align:middle;'>"
                    "<svg width='18' height='18' viewBox='0 0 24 24' style='vertical-align:middle;'>"
                    "<circle cx='12' cy='12' r='10' fill='#0072ff' opacity='0.15'/>"
                    "<path d='M12 6v6l4 2' stroke='#0072ff' stroke-width='2' fill='none' stroke-linecap='round'/>"
                    "<circle cx='12' cy='12' r='2' fill='#0072ff'/>"
                    "</svg>"
                    f"</span>{avg_speed:.2f} km/h</td>"
                )
                report_html += f"<td>{info['frames']}</td>"
                report_html += f"<td><span class='{badge_class}'>{possession}</span></td>"
                report_html += f"</tr>"
            report_html += "</table></div>"
        report_html += """
            </div>
        </body>
        </html>
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        processing_tasks[processing_id]['report_filename'] = report_filename
        processing_tasks[processing_id]['progress'] = 100
        processing_tasks[processing_id]['status'] = 'completed'
        
        # --- Build a team_color_map for stable team coloring in video and report ---
        team_color_map = {}
        for pid, info in player_summary.items():
            team = info['team']
            # Force Team 1 to white, Team 2 to green
            if str(team).lower() == 'team 1':
                color = (255,255,255)
            elif str(team).lower() == 'team 2':
                color = (0,255,0)
            else:
                color = info['team_color']
            if team not in team_color_map:
                team_color_map[team] = color
            # Also update the player's color for the report
            player_summary[pid]['team_color'] = color
        # --- Use stable team assignment and color in video overlays ---
        output_video_frames = tracker.draw_annotations(
            video_frames, tracks, team_ball_control,
            player_team_map=player_team_map,
            team_color_map=team_color_map
        )
        camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
        
        save_video(output_video_frames, output_path)
        
    except Exception as e:
        processing_tasks[processing_id]['status'] = 'error'
        processing_tasks[processing_id]['error_message'] = str(e)


# Original process video function (kept for compatibility)
def process_video(input_path, output_path):
    # This function now just calls the progress-tracking version with a dummy ID
    dummy_id = str(uuid.uuid4())
    processing_tasks[dummy_id] = {
        'progress': 0,
        'status': 'processing',
        'input_path': input_path,
        'output_path': output_path
    }
    process_video_with_progress(dummy_id, input_path, output_path)


# Endpoint to view the report in browser
@app.route('/view_report/<processing_id>')
def view_report(processing_id):
    task = processing_tasks.get(processing_id)
    if not task or 'report_filename' not in task:
        return 'Report not found or not ready.', 404
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], task['report_filename'])
    if not os.path.exists(report_path):
        return 'Report file missing.', 404
    with open(report_path, 'r') as f:
        return f.read()

# Endpoint to download the report
@app.route('/download_report/<processing_id>')
def download_report(processing_id):
    task = processing_tasks.get(processing_id)
    if not task or 'report_filename' not in task:
        return 'Report not found or not ready.', 404
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], task['report_filename'])
    if not os.path.exists(report_path):
        return 'Report file missing.', 404
    # Generate PDF from HTML using pdfkit with explicit wkhtmltopdf path
    pdf_path = report_path.replace('.html', '.pdf')
    config = pdfkit.configuration(wkhtmltopdf='C:/wkhtmltox/bin/wkhtmltopdf.exe')
    pdfkit.from_file(report_path, pdf_path, configuration=config)
    return send_file(pdf_path, as_attachment=True, download_name=f'report_{processing_id}.pdf')

# Run the App
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
