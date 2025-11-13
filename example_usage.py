"""Example usage of People Flow application"""

from peopleflow.app import create_app

# Create the Flask app
app = create_app()

if __name__ == "__main__":
    # Run the application
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True, use_reloader=False)

