from apscheduler.schedulers.background import BackgroundScheduler
import atexit

scheduler = BackgroundScheduler()

def retrain_model():
    # Code to retrain the model with new data
    pass

# Schedule retraining every day at midnight
scheduler.add_job(func=retrain_model, trigger='cron', hour=0)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())
