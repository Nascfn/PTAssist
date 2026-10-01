# Infrastructure

Our Azure resources are created by hand in the Azure Portal. This file is the written record of every setting, so we can rebuild everything in a new subscription (the student credit expires around late February 2027).

**If you change anything in the Portal, update this file in the same PR or right after.**

## Resources

Filled in during Sprint 2 (task SCRUM-46).

| Resource | Azure service | Name | Key settings |
| --- | --- | --- | --- |
| API | Container Apps | | Public HTTPS, scale to zero |
| AI service | Container Apps | | Internal ingress only, scale to zero |
| Database | PostgreSQL Flexible Server (B1ms) | | TLS required, firewall on |
| Video storage | Storage account (Blob) | | Private container, 30-day delete rule |
| Frontend | Static Web Apps | | |
| Migrations | Container Apps job | | Runs Alembic on each deploy |

## Deploys

- **Pull requests:** GitHub Actions runs lint, tests, and image builds.
- **Merge to main:** build and push images to GitHub Container Registry, run migrations, deploy new revisions. Login to Azure uses OIDC (no stored passwords).

## Cost

Budget alerts at $5 and $10. Everything scales to zero when idle.
