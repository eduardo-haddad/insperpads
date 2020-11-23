from flask import Blueprint, request, make_response
from src.main import main

# Blueprint object
blueprint = Blueprint(name=__name__, import_name=__name__)


@blueprint.route("/main", methods=["GET", "POST"])
def main():
    try:
        # Request
        req = request.get_json()

        # Run main function
        main()

    except Exception as err:
        # 401 error response
        # Logging and status set in rq.error_handlers.py
        return ("{}\n".format(str(err)), 401)
    else:
        # 200 success response
        return ("OK!\n", 200)

