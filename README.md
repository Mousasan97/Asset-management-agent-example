# OpenAI API Integration

This project provides a Flask-based web application that integrates with the OpenAI API to provide various AI-powered functionalities.

## Features

- OpenAI API integration
- RESTful API endpoints
- Comprehensive logging system
- Environment-based configuration
- Error handling and validation

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
```
Then edit the `.env` file with your actual configuration values.

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and update the values:

- `OPENAI_API_KEY`: Your OpenAI API key
- `FLASK_APP`: The main application file (default: app.py)
- `FLASK_ENV`: Environment (development/production)
- `FLASK_DEBUG`: Debug mode (1/0)
- `HOST`: Server host
- `PORT`: Server port
- `LOG_LEVEL`: Logging level
- `LOG_FILE`: Log file path
- `SECRET_KEY`: Flask secret key

## Usage

1. Start the server:
```bash
flask run
```

2. The API will be available at `http://localhost:5000`

## API Endpoints

- `POST /api/chat`: Chat with OpenAI API
- `POST /api/completion`: Get text completion from OpenAI API
- Additional endpoints as documented in the code

## Logging

The application uses a comprehensive logging system:
- Logs are written to the file specified in `LOG_FILE`
- Log level can be configured via `LOG_LEVEL`
- Both file and console logging are supported

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository. 