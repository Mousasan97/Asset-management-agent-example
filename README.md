# Asset Management Agent Example

This project provides a Flask-based web application that integrates with the OpenAI API to create an intelligent asset management system. The application helps in managing and analyzing assets using AI-powered insights and recommendations.

## Features

- OpenAI API integration for intelligent asset analysis
- RESTful API endpoints for asset management
- Comprehensive logging system
- Environment-based configuration
- Error handling and validation
- Asset tracking and management capabilities
- AI-powered insights and recommendations

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mousasan97/Asset-management-agent-example.git
cd Asset-management-agent-example
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

- `OPENAI_API_KEY`: Your OpenAI API key (required for the AI assistant functionality)
  - You can obtain an API key from [OpenAI's platform](https://platform.openai.com/account/api-keys)
  - This is required for the application to start properly
- `FLASK_APP`: The main application file (default: app.py)
- `FLASK_ENV`: Environment (development/production)
- `FLASK_DEBUG`: Debug mode (1/0)
- `HOST`: Server host
- `PORT`: Server port
- `LOG_LEVEL`: Logging level
- `LOG_FILE`: Log file path
- `SECRET_KEY`: Flask secret key
- `API_URL`: URL for the Assets API
- `ACTIVITIES_API_URL`: URL for the Activities API
- `AUTH_KEY` and `AUTH_SECRET`: Authentication credentials for the Asset Management API

## Usage

1. Start the server:
```bash
flask run
```

2. The API will be available at `http://localhost:5000`

## API Endpoints

- `POST /api/chat`: Chat with OpenAI API for asset-related queries
- `POST /api/completion`: Get AI-powered insights and recommendations for assets
- Additional endpoints as documented in the code

## Logging

The application uses a comprehensive logging system:
- Logs are written to the file specified in `LOG_FILE`
- Log level can be configured via `LOG_LEVEL`
- Both file and console logging are supported

## Troubleshooting

If you encounter an error like:
```
CRITICAL: Failed to create OpenAI Assistant during startup: Error code: 401 - {'error': {'message': 'Incorrect API key provided...}}
```

This means your OpenAI API key is not valid. Make sure to:
1. Obtain a valid API key from [OpenAI's platform](https://platform.openai.com/account/api-keys)
2. Update the `OPENAI_API_KEY` value in your `.env` file with the actual key

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