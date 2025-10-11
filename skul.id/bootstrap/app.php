<?php

use Illuminate\Foundation\Application;
use Illuminate\Session\Middleware\StartSession;
use Illuminate\Foundation\Configuration\Exceptions;
use Illuminate\Foundation\Configuration\Middleware;

return Application::configure(basePath: dirname(__DIR__))
    ->withRouting(
        web: __DIR__ . '/../routes/web.php',
        commands: __DIR__ . '/../routes/console.php',
        health: '/up',
    )
    ->withMiddleware(function (Middleware $middleware) {
        $middleware->alias([
            'role' => App\Http\Middleware\Role::class,
            'auth' => App\Http\Middleware\Authenticate::class,
            'profile' => App\Http\Middleware\ProfileComplete::class,
            'prevent-back-history' => App\Http\Middleware\PreventBackHistory::class,
            'guest.redirect' => App\Http\Middleware\RedirectIfAuthenticatedCustom::class,
            'count-website-visit' => App\Http\Middleware\CountWebsiteVisit::class,
        ]);

        // $middleware->append([
        //     StartSession::class,
        //     App\Http\Middleware\CountWebsiteVisit::class,
        // ]);
    })
    ->withExceptions(function (Exceptions $exceptions) {
        //
    })->create();
