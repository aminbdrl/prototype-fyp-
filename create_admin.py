from werkzeug.security import generate_password_hash
from app import app, db, AdminUser

with app.app_context():

    existing = AdminUser.query.filter_by(username="admin").first()

    if not existing:
        admin = AdminUser(
            username="admin",
            password=generate_password_hash("admin123")
        )

        db.session.add(admin)
        db.session.commit()

        print("Admin created successfully.")

    else:
        print("Admin already exists.")