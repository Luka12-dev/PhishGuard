import sys
import os
import re
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QFileDialog, QProgressBar, QMessageBox, QLineEdit,
    QCheckBox, QGroupBox, QGridLayout, QSizePolicy, QSpacerItem
)
from PyQt6.QtGui import QAction, QKeySequence, QShortcut, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

# QSS (polished)
QSS = r"""
QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #071029, stop:1 #071a2f); color: #e6eef8; }
QLabel#Title { font-size: 20px; font-weight:600; color: #f8fafc; }
QLabel { color: #e6eef8; }
QPushButton { background: #0f1724; color: #e6eef8; padding: 8px 12px; border-radius:8px; border: 1px solid #112233; }
QPushButton:hover { background: #111827; }
QLineEdit, QTextEdit { background: #071828; color: #e6eef8; border: 1px solid #0f3150; padding: 8px; border-radius:6px; }
QCheckBox { color: #e6eef8; }
QProgressBar { background: #03101a; color: #e6eef8; border-radius:8px; height: 14px; }
QGroupBox { border: 1px solid #082233; margin-top: 6px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
#statusBadge { background: #064e3b; color: #d1fae5; padding: 4px 8px; border-radius: 10px; font-weight:600; }
#statusBadge.off { background: #4c1d1d; color: #ffd7d7; }
"""

URL_REGEX = re.compile(r'https?://[\w\-._~:/?#[\]@!$&'""" + "()*+,;=%]+""")
URL_REGEX = re.compile(r'https?://[\w./?=&%\-_:~#]+', flags=re.IGNORECASE)
EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+', flags=re.IGNORECASE)
NON_ALNUM = re.compile(r'[^a-z0-9\s]')

def extract_urls(text):
    return URL_REGEX.findall(text)

def extract_emails(text):
    return EMAIL_REGEX.findall(text)

def preprocess(text: str) -> str:
    t = text.lower()
    # keep urls/emails for extraction but replace them for vectorizer
    t = re.sub(r'http\S+',' ', t)
    t = NON_ALNUM.sub(' ', t)
    t = re.sub(r'\s+',' ', t).strip()
    return t

# Worker threads
class TrainerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def run(self):
        try:
            self.progress.emit(5)
            texts = [preprocess(t) for t in self.examples['text']]
            y = self.examples['label']
            vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
            clf = LogisticRegression(max_iter=1000)
            self.progress.emit(25)
            pipe = Pipeline([('vec', vec), ('clf', clf)])
            pipe.fit(texts, y)
            self.progress.emit(90)
            self.finished.emit(pipe)
            self.progress.emit(100)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)

class PredictorThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pipeline, text):
        super().__init__()
        self.pipeline = pipeline
        self.text = text

    def run(self):
        try:
            X = [preprocess(self.text)]
            probs = self.pipeline.predict_proba(X)
            score = float(probs[0][1])  # probability of phishing
            # explainability
            vec = self.pipeline.named_steps['vec']
            clf = self.pipeline.named_steps['clf']
            Xv = vec.transform(X)
            feature_names = vec.get_feature_names_out()
            arr = Xv.toarray()[0]
            # contribution = arr * coef (positive contribution to class 1)
            contrib = arr * (clf.coef_[0])
            top_idx = np.argsort(-np.abs(contrib))[:12]
            tokens = [(feature_names[i], float(contrib[i])) for i in top_idx if arr[i] > 0 or abs(contrib[i])>0]
            result = {'score': score, 'tokens': tokens}
            self.finished.emit(result)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)

# Synthetic dataset builder
def build_synthetic_examples():
    phish = [
        "verify your account immediately using this link",
        "your account will be suspended without confirmation",
        "we detected unusual activity please login now",
        "payment required update billing immediately via link",
        "invoice attached please download and complete payment",
        "urgent confirm identity to prevent account suspension",
        "password expired reset immediately using this link",
        "suspicious transaction detected confirm payment details now",
        "secure message awaiting open link to view",
        "mailbox storage full confirm details to continue",
        "refund pending provide banking details to receive",
        "verify two factor code at this link",
        "unauthorized login detected confirm your identity now",
        "download attachment and submit required personal information",
        "confirm payment method to avoid service interruption",
        "click link to accept confidential document now",
        "account access limited verify credentials to restore",
        "security team requires identity confirmation before release",
        "we attempted charge please confirm your card",
        "confirm recent purchase to prevent fraudulent charge",
        "complete verification form to unlock your account",
        "verify account ownership to avoid data deletion",
        "alert payment failed update billing details immediately",
        "confirm shipping address to process your order",
        "security update required install via enclosed attachment",
        "your benefits require confirmation complete form today",
        "link expires soon verify now to retain",
        "verify identity to recover lost account access",
        "confirm bank details to process refund today",
        "sensitive document requires authentication before download now",
        "unauthorized purchase blocked confirm shipping to proceed",
        "verify account security questions to continue now",
        "email verification needed to access important files",
        "confirm identity to receive urgent legal notice",
        "update payment info to prevent cancellation tomorrow",
        "we noticed unfamiliar device confirm it now",
        "secure portal message requires verification before viewing",
        "click to verify email address restore access",
        "confirm payment authorization to avoid collection actions",
        "your identity must be verified to release",
        "verify billing address for successful order fulfillment",
        "account flagged for review provide verification immediately",
        "confirm tax information to receive government refund",
        "reset password now to secure your profile",
        "validate account details to enable new services",
        "urgent verification required click the link below",
        "your account shows suspicious logins please respond",
        "confirm identity before transferring funds to account",
        "verify personal information to complete registration process",
        "authentication needed to access transmitted secure documents"
    ]

    legit = [
        "meeting agenda next week attached please review",
        "project update deliverables timeline responsibilities and deadlines",
        "family reunion photos uploaded to shared drive",
        "your order has shipped tracking number included",
        "monthly newsletter includes product updates and tips",
        "please review minutes from yesterday meeting attached",
        "flight itinerary with dates times and confirmation",
        "deliverables due next Friday please confirm availability",
        "HR updated vacation policy available for employees",
        "research articles relevant to thesis uploaded online",
        "subscription renewal succeeded no action is required",
        "scheduled one hour project status check in",
        "team lunch planned at downtown cafe Tuesday",
        "invoice for last months services has processed",
        "appointment confirmation and location details are included",
        "thank you for your purchase receipt is attached",
        "please RSVP for company retreat by Friday",
        "shared document contains reviewer feedback please check",
        "package delivered please confirm receipt at address",
        "class materials homework assignments uploaded to portal",
        "project timeline updated based on client feedback",
        "weekly report summarizes key metrics and progress",
        "join the video call using this link",
        "family calendar updated with birthdays and events",
        "account statement for last quarter attached records",
        "conference schedule speaker list and venue attached",
        "thank you for attending please complete survey",
        "volunteer sign up sheet open for event",
        "troubleshooting steps for your device are attached",
        "reservation at the hotel next month confirmed",
        "course syllabus grading rubric and schedule uploaded",
        "password change successful no further action required",
        "annual report attached for board review consideration",
        "congratulations team for reaching milestone well done",
        "warranty registration confirmation attached keep for reference",
        "library uploaded requested copies to shared folder",
        "monthly maintenance scheduled Saturday brief interruption expected",
        "assigned mentor will contact you within days",
        "software license key and download instructions included",
        "doctor confirmed appointment Tuesday at ten",
        "additional resources and slides from presentation attached",
        "community newsletter highlights workshops classes and news",
        "seller shipped replacement part track delivery online",
        "onboarding session scheduled required forms attached complete",
        "classroom substitution plan and seating chart attached",
        "application status updated check portal for details",
        "review attached contract sign if you agree",
        "team building activities scheduled next Friday morning",
        "please find the invoice attached for payment",
        "training materials uploaded complete modules before session"
    ]
    data = {'text': phish + legit, 'label': [1]*len(phish) + [0]*len(legit)}
    return pd.DataFrame(data)

# Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PhishGuard')
        self.setWindowIcon(QIcon("AI3.ico"))
        self.resize(1100, 700)
        self.pipeline = None
        self._predictor_thread = None
        self._trainer_thread = None

        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(); central.setLayout(outer)

        # Header
        header = QHBoxLayout()
        title = QLabel('PhishGuard')
        title.setObjectName('Title')
        subtitle = QLabel('Real-time phishing detection')
        subtitle.setStyleSheet('color: #cfe8ff;')
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()
        self.statusBadge = QLabel('No model')
        self.statusBadge.setObjectName('statusBadge')
        self.statusBadge.setProperty('class', 'off')
        header.addWidget(self.statusBadge)
        outer.addLayout(header)

        # Main content layout
        main = QHBoxLayout()
        outer.addLayout(main)

        # Left column: Inputs & controls
        left = QVBoxLayout()
        main.addLayout(left, 2)

        # Instruction box
        instr = QGroupBox('Quick instructions & examples')
        instr_layout = QVBoxLayout(); instr.setLayout(instr_layout)
        instr_layout.addWidget(QLabel('Paste an email subject and body (or a URL) into the input field below.'))
        instr_layout.addWidget(QLabel('Examples to test (copy/paste):'))
        ex = QLabel("• Phishing: 'Verify your account now! Click this link to update your password.'\n• Legit: 'Meeting agenda for Monday attached. Please review and reply.'")
        ex.setStyleSheet('color:#dbeafe;')
        instr_layout.addWidget(ex)
        left.addWidget(instr)

        # Inputs
        lbl_subject = QLabel('Subject (optional)')
        self.input_subject = QLineEdit(); self.input_subject.setPlaceholderText('e.g. Verify your account')
        left.addWidget(lbl_subject); left.addWidget(self.input_subject)

        lbl_from = QLabel('From (optional)')
        self.input_from = QLineEdit(); self.input_from.setPlaceholderText('e.g. support@yourbank.com')
        left.addWidget(lbl_from); left.addWidget(self.input_from)

        lbl_body = QLabel('Email body or URL (required)')
        self.input_body = QTextEdit(); self.input_body.setPlaceholderText('Paste full message or URL here. For best results include message context and any links.'); self.input_body.setFixedHeight(260)
        left.addWidget(lbl_body); left.addWidget(self.input_body)

        # Controls row
        controls = QHBoxLayout()
        self.btn_analyze = QPushButton('Analyze Now')
        self.btn_analyze.clicked.connect(self.on_analyze_now)
        self.btn_train = QPushButton('Train Demo Model')
        self.btn_train.clicked.connect(self.on_train_demo)
        self.btn_load = QPushButton('Load Model')
        self.btn_load.clicked.connect(self.on_load_model)
        self.btn_save = QPushButton('Save Model')
        self.btn_save.clicked.connect(self.on_save_model)
        controls.addWidget(self.btn_analyze); controls.addWidget(self.btn_train); controls.addWidget(self.btn_load); controls.addWidget(self.btn_save)
        left.addLayout(controls)

        # Real-time toggle and debounce hint
        rt_layout = QHBoxLayout()
        self.rt_checkbox = QCheckBox('Real-time (debounced 700 ms)')
        self.rt_checkbox.setChecked(False)
        self.rt_checkbox.stateChanged.connect(self.on_realtime_toggled)
        rt_layout.addWidget(self.rt_checkbox)
        rt_layout.addStretch()
        left.addLayout(rt_layout)

        # Log / progress
        self.progress = QProgressBar(); self.progress.setValue(0)
        left.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(120)
        left.addWidget(self.log)

        # Right column: Results & explainability
        right = QVBoxLayout()
        main.addLayout(right, 2)

        self.result_title = QLabel('Result: (no analysis yet)')
        self.result_title.setStyleSheet('font-weight:600; font-size:14px;')
        right.addWidget(self.result_title)

        self.score_label = QLabel('Confidence: N/A')
        right.addWidget(self.score_label)

        self.tokens_box = QTextEdit(); self.tokens_box.setReadOnly(True); self.tokens_box.setFixedHeight(180)
        right.addWidget(QLabel('Top token contributions'))
        right.addWidget(self.tokens_box)

        self.extracted_box = QTextEdit(); self.extracted_box.setReadOnly(True); self.extracted_box.setFixedHeight(120)
        right.addWidget(QLabel('Extracted URLs and emails'))
        right.addWidget(self.extracted_box)

        # Bottom actions
        bottom = QHBoxLayout()
        self.btn_clear = QPushButton('Clear Inputs'); self.btn_clear.clicked.connect(self.on_clear)
        self.btn_report = QPushButton('Export Quick Report'); self.btn_report.clicked.connect(self.on_export_report)
        bottom.addWidget(self.btn_clear); bottom.addWidget(self.btn_report)
        right.addLayout(bottom)

        # Spacing
        outer.addItem(QSpacerItem(0,8))

        # Timer for debounce real-time
        self._debounce_timer = QTimer(); self._debounce_timer.setSingleShot(True); self._debounce_timer.timeout.connect(self._on_debounce_timeout)
        # connect text edits
        self.input_subject.textChanged.connect(self._maybe_debounce)
        self.input_body.textChanged.connect(self._maybe_debounce)

        # Shortcut F11 for fullscreen
        QShortcut(QKeySequence('F11'), self, activated=self.toggle_fullscreen)

        # apply style
        self.setStyleSheet(QSS)

        # Build an initial demo model so user can analyze immediately
        self._build_initial_model()
        self.log_msg('Ready - demo model loaded. Use "Train Demo Model" for a fresh model or load your own.')

    # UI helpers
    def log_msg(self, s: str):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log.append(f'[{ts}] {s}')

    def set_status(self, text: str, ok: bool = True):
        self.statusBadge.setText(text)
        if ok:
            self.statusBadge.setProperty('class', '')
            self.statusBadge.setStyleSheet('background: #064e3b; color: #d1fae5; padding:4px 8px; border-radius:10px;')
        else:
            self.statusBadge.setProperty('class', 'off')
            self.statusBadge.setStyleSheet('background: #4c1d1d; color: #ffd7d7; padding:4px 8px; border-radius:10px;')

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal(); self.log_msg('Exited fullscreen')
        else:
            self.showFullScreen(); self.log_msg('Entered fullscreen')

    # Debounce real-time
    def on_realtime_toggled(self):
        if self.rt_checkbox.isChecked():
            self.log_msg('Real-time enabled — edits will trigger analysis after a short pause')
        else:
            self.log_msg('Real-time disabled')

    def _maybe_debounce(self):
        if self.rt_checkbox.isChecked():
            self._debounce_timer.start(700)

    def _on_debounce_timeout(self):
        self.on_analyze_now()

    # Analysis flow
    def _build_initial_model(self):
        examples = build_synthetic_examples()
        # train in main thread quickly (small dataset) — keep progress bar minimal
        try:
            texts = [preprocess(t) for t in examples['text']]
            y = examples['label']
            vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
            clf = LogisticRegression(max_iter=1000)
            pipe = Pipeline([('vec', vec), ('clf', clf)])
            pipe.fit(texts, y)
            self.pipeline = pipe
            self.set_status('Demo model loaded', ok=True)
        except Exception as e:
            self.set_status('Model error', ok=False)
            self.log_msg('Initial model build failed: ' + str(e))

    def on_train_demo(self):
        examples = build_synthetic_examples()
        self._trainer_thread = TrainerThread(examples)
        self._trainer_thread.progress.connect(self.progress.setValue)
        self._trainer_thread.finished.connect(self._on_trained)
        self._trainer_thread.error.connect(self._on_worker_error)
        self._trainer_thread.start()
        self.set_status('Training...', ok=True)
        self.log_msg('Training demo model in background...')

    def _on_trained(self, pipeline):
        self.pipeline = pipeline
        self.progress.setValue(0)
        self.set_status('Trained model', ok=True)
        self.log_msg('Training finished — model is ready')

    def on_analyze_now(self):
        text = (self.input_subject.text().strip() + ' ' + self.input_body.toPlainText().strip()).strip()
        if not text:
            QMessageBox.warning(self, 'Input required', 'Please paste an email body or URL to analyze.')
            return
        if self.pipeline is None:
            QMessageBox.warning(self, 'No model', 'No model is loaded. Train or load a model first.')
            return
        # extract urls and emails
        urls = extract_urls(text)
        emails = extract_emails(text)
        self.extracted_box.setPlainText('URLs:\n' + '\n'.join(urls) + '\n\nEmails:\n' + '\n'.join(emails))

        # start predictor thread
        self.btn_analyze.setEnabled(False)
        self._predictor_thread = PredictorThread(self.pipeline, text)
        self._predictor_thread.finished.connect(self._on_prediction)
        self._predictor_thread.error.connect(self._on_worker_error)
        self._predictor_thread.start()
        self.log_msg('Analysis started...')

    def _on_prediction(self, res):
        try:
            score = res.get('score', 0.0)
            tokens = res.get('tokens', [])
            label = 'PHISHING' if score >= 0.5 else 'LEGITIMATE'
            pct = round(score*100, 2)
            self.result_title.setText(f'Result: {label}')
            self.score_label.setText(f'Confidence (phishing): {pct}%')
            # tokens presentation
            lines = []
            for t, c in tokens:
                sign = '+' if c>0 else '-'
                lines.append(f"{t}  ({sign}{abs(round(c,4))})")
            self.tokens_box.setPlainText('\n'.join(lines) if lines else 'No strong token contributions detected')
            self.set_status('Model ready', ok=True)
            self.log_msg(f'Analysis finished — {label} ({pct}%)')
        except Exception as e:
            self.log_msg('Prediction handling error: ' + str(e))
        finally:
            self.btn_analyze.setEnabled(True)

    def _on_worker_error(self, tb):
        self.set_status('Worker error', ok=False)
        QMessageBox.critical(self, 'Worker error', 'A background operation failed — see log for details')
        self.log_msg('Worker thread failed:\n' + str(tb))

    # File / model operations
    def on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load model (.joblib)', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        try:
            obj = joblib.load(path)
            self.pipeline = obj
            self.set_status('Model loaded', ok=True)
            self.log_msg(f'Loaded model from: {path}')
        except Exception as e:
            self.set_status('Load failed', ok=False)
            QMessageBox.critical(self, 'Load error', str(e))
            self.log_msg('Load model failed: ' + str(e))

    def on_save_model(self):
        if self.pipeline is None:
            QMessageBox.warning(self, 'No model', 'There is no model to save.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save model (.joblib)', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        try:
            joblib.dump(self.pipeline, path)
            self.log_msg(f'Model saved to: {path}')
            QMessageBox.information(self, 'Saved', 'Model saved successfully.')
        except Exception as e:
            QMessageBox.critical(self, 'Save error', str(e))
            self.log_msg('Save failed: ' + str(e))

    def on_clear(self):
        self.input_subject.clear(); self.input_body.clear(); self.extracted_box.clear(); self.tokens_box.clear(); self.result_title.setText('Result: (no analysis yet)'); self.score_label.setText('Confidence: N/A')

    def on_export_report(self):
        text = f"Timestamp: {datetime.now().isoformat()}\nResult: {self.result_title.text()}\n{self.score_label.text()}\n\nTop tokens:\n{self.tokens_box.toPlainText()}\n\nExtracted:\n{self.extracted_box.toPlainText()}\n"
        path, _ = QFileDialog.getSaveFileName(self, 'Export quick report (.txt)', os.getcwd(), 'Text Files (*.txt)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            QMessageBox.information(self, 'Exported', f'Report saved to {path}')
            self.log_msg('Report exported to: ' + path)
        except Exception as e:
            QMessageBox.critical(self, 'Export error', str(e))
            self.log_msg('Export failed: ' + str(e))

# Main entrypoint
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())