![](/docs/assets/banner.png)

# Cohere Toolkit

Toolkit is a deployable all-in-one RAG application that enables users to quickly build their LLM-based product.

- [Try Toolkit](#try-now)
- [About Toolkit](#about-toolkit)
- [Toolkit Setup](/docs/setup.md)
- [Troubleshooting](/docs/troubleshooting.md)
- [How to guides](/docs/how_to_guides.md)
  - [How to set up command model providers](/docs/command_model_providers.md)
  - [How to add tools](/docs/custom_tool_guides/tool_guide.md)
  - [How to add auth to your tools](/docs/custom_tool_guides/tool_auth_guide.md)
  - [How to setup Google Drive](/docs/custom_tool_guides/google_drive.md)
  - [How to add authentication](/docs/auth_guide.md)
  - [How to deploy toolkit services](/docs/service_deployments.md)
  - [How to set up Github Actions for automated DB migrations](/docs/github_migrations_action.md)
  - [How to customize the theme](/docs/theming.md)
  - [How to contribute](#contributing)
- [Try Cohere's Command Showcase](https://coral.cohere.com/)

![](/docs/assets/toolkit.gif)

## Try Now:

*You will need to have [Docker](https://www.docker.com/products/docker-desktop/) and [Docker-compose >= 2.22](https://docs.docker.com/compose/install/) installed. [Go here for a more detailed setup.](/docs/setup.md)*

There are two main ways of running Toolkit:

1. **(Recommended) Docker Compose setup**

Note: to include community tools when building locally, set the `INSTALL_COMMUNITY_DEPS` build arg in the `docker-compose.yml` to `true`.

```bash
git clone https://github.com/cohere-ai/cohere-toolkit.git
cd cohere-toolkit
make first-run
```

2. **Docker Image setup**

We offer pre-built Docker images as well that you can run, but note that they might contain out-of-date code, so the local compose setup is preferred.

```bash
docker run -e COHERE_API_KEY='>>YOUR_API_KEY<<' -p 8000:8000 -p 4000:4000 ghcr.io/cohere-ai/cohere-toolkit:latest
```

Go to [localhost:4000](http://localhost:4000/) in your browser and start chatting with Cohere's Command model!

## GitHub Codespaces

To run this project using GitHub Codespaces, please refer to our [Codespaces Setup Guide](/docs/github_codespaces.md).

## About Toolkit

![](/docs/assets/toolkit_graphic.png)

- **Interfaces** - any client-side UI, currently contains a web app and Slack bot.
  - Defaults to Cohere's Web UI at `src/interfaces/coral_web` - A web app built in Next.js. Includes a simple SQL database out of the box to store conversation history in the app.
  - You can turn on the Assistants view that uses `src/interfaces/assistants_web` instead.
- **Backend API** - in `src/backend` this follows a similar structure to the [Cohere Chat API](https://docs.cohere.com/reference/chat) but also include customizable elements: 
  - **Model** - you can customize with which provider you access Cohere's Command models. By default included in the toolkit is Cohere's Platform, Sagemaker, Azure, Bedrock, HuggingFace, local models. [More details here.](/docs/command_model_providers.md)
  - **Retrieval**- you can customize tools and data sources that the application is run with.
- **Service Deployment Guides** - we also include guides for how to deploy the toolkit services in production including with AWS, GCP and Azure. [More details here.](/docs/service_deployments.md)

## Contributing

Contributions are what drive an open source community, any contributions made are greatly appreciated. To get started, check out our [documentation.](CONTRIBUTING.md)

## Contributors

<a href="https://github.com/cohere-ai/cohere-toolkit/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cohere-ai/cohere-toolkit" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
