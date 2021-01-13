# Contributing to UKIS-csmask

Welcome to `UKIS-csmask`. We invite anyone to participate by contributing code, reporting bugs, fixing bugs, writing documentation or discussing the future of this project. As a contributor, here are the guidelines we would like you to follow:

 - [Code of Conduct](#coc)
 - [Issues and Bugs](#issue)
 - [Feature Requests](#feature)
 - [Code Conventions](#rules)
 - [Submission Guidelines](#submit)
 - [Signing the CLA](#cla)

## <a name="coc"></a> Code of Conduct
Help us keep `UKIS-csmask` open and inclusive. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## <a name="issue"></a> Found a Bug?
If you find a bug in the source code, you can help us by
[submitting an issue](#submit-issue) to our [GitHub Repository](https://github.com/dlr-eoc/). Even better, you can
[submit a Pull Request](#submit-pr) with a fix.

## <a name="feature"></a> Missing a Feature?
You can *request* a new feature by [submitting an issue](#submit-issue) to our GitHub
Repository. If you would like to *implement* a new feature, please submit an issue with
a proposal for your work first, to be sure that we can use it.

## <a name="rules"></a> Code Conventions
To ensure consistency throughout the source code, keep these rules in mind as you are working:

* All features or bug fixes **must be tested** by one or more specs (unit-tests).
* Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
* We use a line-length of 120 and [black](https://github.com/psf/black) to format our code.

## <a name="submit"></a> Submission Guidelines

### <a name="submit-issue"></a> Submitting an Issue

Before you submit an issue, please search the issue tracker, maybe an issue for your problem already exists and the discussion might inform you of workarounds readily available.

We want to fix all the issues as soon as possible, but before fixing a bug we need to reproduce and confirm it. In order to reproduce bugs, we will systematically ask you to provide a minimal reproduction. Having a minimal reproducible scenario gives us a wealth of important information without going back & forth to you with additional questions.

A minimal reproduction allows us to quickly confirm a bug (or point out a coding problem) as well as confirm that we are fixing the right problem.

### <a name="submit-pr"></a> Submitting a Pull Request (PR)
Before you submit your Pull Request (PR) consider the following guidelines:

1. Search GitHub for an open or closed PR that relates to your submission. You don't want to duplicate effort.
1. Be sure that an issue describes the problem you're fixing, or documents the design for the feature you'd like to add.
  Discussing the design up front helps to ensure that we're ready to accept your work.
1. Please sign our [Contributor License Agreement (CLA)](#cla) before sending PRs.
  We cannot accept code without this. Make sure you sign with the primary email address of the Git identity that has been granted access to the UKIS repository.
1. Fork the UKIS repo.
1. Make your changes in a new git branch:

     ```shell
     git checkout -b my-fix-branch master
     ```

1. Create your patch, **including appropriate test cases**.
1. Document your changes in the [changelog](CHANGELOG.rst).
    
     ```shell
     git commit -a -m "some useful message"
     ```
    Note: the optional commit `-a` command line option will automatically "add" and "rm" edited files.

1. Push your branch to GitHub:
   
    ```shell
    git push origin my-fix-branch
    ```

1. In GitHub, send a pull request to `master`.
* If we suggest changes then:
  * Make the required updates.
  * Re-run the tests to ensure they are still passing.
  * Rebase your branch and force push to your GitHub repository (this will update your Pull Request):

    ```shell
    git rebase master -i
    git push -f
    ```

That's it! Thank you for your contribution!


## <a name="changelogGuidelines"></a> Changelog guidelines

 - Document your changes at the very top of the file.
 - Categorize your changes as one of
   - Features (*Added*)
   - Bug Fixes (*Fix*)
   - Other changes (*Changed*)
 - For each change, add one item containing
   - The module/project changed (not required for 'other changes')
   - A short description of the change


## <a name="cla"></a> Signing the CLA

Please sign our Contributor License Agreement (CLA) before sending pull requests. For any code
changes to be accepted, the CLA must be signed. It's a quick process, we promise! We'll need you to
  [print, sign and one of scan+email, fax or mail the form](DLR_Individual_Contributor_License_Agreement_UKIS.pdf).

<hr>

  If you have more than one Git identity, you must make sure that you sign the CLA using the primary email address associated with the ID that has been granted access to the UKIS repository. Git identities can be associated with more than one email address, and only one is primary. Here are some links to help you sort out multiple Git identities and email addresses:

  * https://help.github.com/articles/setting-your-commit-email-address-in-git/
  * https://stackoverflow.com/questions/37245303/what-does-usera-committed-with-userb-13-days-ago-on-github-mean
  * https://help.github.com/articles/about-commit-email-addresses/
  * https://help.github.com/articles/blocking-command-line-pushes-that-expose-your-personal-email-address/

  Note that if you have more than one Git identity, it is important to verify that you are logged in with the same ID with which you signed the CLA, before you commit changes. If not, your PR will fail the CLA check.

<hr>
